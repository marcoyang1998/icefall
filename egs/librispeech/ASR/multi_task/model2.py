# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import random
import warnings
from typing import Dict, Optional, Tuple

import k2
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from asp import AttentiveStatisticsPooling
from encoder_interface import EncoderInterface
from scaling import ScaledLinear, penalize_abs_values_gt, SwooshR
from icefall.utils import add_sos, make_pad_mask


class PromptedAudioEncoder(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        vocab_size: int,
        use_soft_prompt: bool = False,
        num_tasks: int = 5,
        soft_prompt_len: int = 10,
        soft_prompt_dim: int = 80,
        universal_prompt_prob: float=0.1,
        do_ASR: bool = True,
        do_AT: bool = True,
        do_SV: bool = True,
        sv_KD: bool = False,
        num_spkrs: int = 2377,
        speaker_embed_dim: int = 192,
    ):
        """
        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          text_encoder:
            This is a encoder that processes text information (e.g content prompt
            and style prompt). The input is `x` of (N,T) and `x_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
          text_encoder_type:
            The type of the text_encoder. Supported are (BERT, DistilBERT)
          context_fuser
            A optional module that fuses the embeddings of text encoder. The fused embedding
            will be added to the joiner.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        # assert hasattr(decoder, "blank_id")

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        
        self.do_ASR = do_ASR
        if self.do_ASR:
            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim,
                vocab_size,
                initial_scale=0.25,
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim,
                vocab_size,
                initial_scale=0.25,
            )

        self.num_tasks = num_tasks
        self.soft_prompt_dim = soft_prompt_dim
        self.soft_prompt_len = soft_prompt_len
        
        self.use_soft_prompt = use_soft_prompt
        # +1 because we also want to have a universal prompt
        # the universal prompt should have ids=0
        if self.use_soft_prompt:
            self.soft_prompt_embed = nn.Embedding((num_tasks+1), soft_prompt_len * soft_prompt_dim)
            self.prompt_proj = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(soft_prompt_dim, encoder_dim),
                SwooshR(),
            )

        # audio tagging
        self.do_AT = do_AT
        if self.do_AT:
            self.audio_tagging_proj = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(encoder_dim, 527),
            ) # 527 classes
        
        # speaker verification
        self.do_SV = do_SV
        self.sv_KD = sv_KD
        if self.do_SV:
            self.speaker_embed_dim = speaker_embed_dim
            self.asp = AttentiveStatisticsPooling(channels=encoder_dim)
            self.speaker_proj = nn.Linear(2 * encoder_dim, speaker_embed_dim)
            self.speaker_pred = nn.Linear(speaker_embed_dim, num_spkrs)
            
            # when performing KD for SV, use cosine similarity as loss function
            # otherwise do the CE-based loss
            if self.sv_KD:
                self.sv_loss_mode = "cosine"
            else:
                self.sv_loss_mode = "ce"
            
        self.universal_prompt_prob = universal_prompt_prob
        
        
    def forward_task_id(
        self, task_ids,
    ):
        """Produce the task specific soft prompt

        Args:
            task_ids (torch.Tensor): task ids (N,)

        Returns:
            soft_prompt embeddings: (L,N,C), L is the soft_prompt length
        """
        N = task_ids.size(0)
        # By p=0.1, use the universal prompt
        # if self.training:
        #     mask = torch.rand(N) < self.universal_prompt_prob
        #     task_ids[mask] = 0
        if random.random() < 0.02:
            logging.info(f"task ids: {task_ids}")
        soft_prompt = self.soft_prompt_embed(task_ids)
        soft_prompt = soft_prompt.reshape(N, self.soft_prompt_len, self.soft_prompt_dim)
        soft_prompt = self.prompt_proj(soft_prompt) # (N, soft_prompt_len, encoder_dim)
        soft_prompt = soft_prompt.permute(1,0,2)
        return soft_prompt

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="none",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="none",
            )

        return simple_loss, pruned_loss

    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        task_ids: torch.Tensor,
    ):
        x, x_lens = self.encoder_embed(x, x_lens)

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # get the task prompts
        if self.use_soft_prompt and task_ids is not None:
            soft_prompt = self.forward_task_id(task_ids) # (N, soft_prompt_len, encoder_dim)
        else:
            soft_prompt = None

        encoder_out, encoder_out_lens = self.encoder(
            x,
            x_lens,
            src_key_padding_mask,
            memory=soft_prompt,
            memory_key_padding_mask=None,
        )
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        assert torch.all(encoder_out_lens > 0)
        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        task_ids: torch.Tensor = None,
        at_targets: torch.Tensor = None,
        sv_targets: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          text:
            A 2-D tensor of integer dtype containing prompt text, of shape (N, T).
            It is exptected to contain the style prompt (first) and then the content
            prompt.
          task_ids:
            The task-id of each training sample (N,)
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0
        device = x.device
        
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens, task_ids)
        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        # ASR loss
        if self.do_ASR:
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)
        
        # audio tagging
        if self.do_AT:
            at_loss = self.forward_audio_tagging(
                encoder_out, encoder_out_lens, at_targets,
            )
        else:
            at_loss = torch.empty(0)
        
        # speaker verification
        if self.do_SV and sv_targets is not None:
            sv_loss = self.forward_speaker_verification(
                encoder_out, encoder_out_lens, target=sv_targets, return_embedding=False, loss_mode=self.sv_loss_mode
            )
        else:
            sv_loss = torch.empty(0)
        
        return simple_loss, pruned_loss, at_loss, sv_loss

    def forward_audio_tagging(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        target: torch.Tensor = None,
        return_logits: bool = False,
    ):
        # target: (N, num_events)
        logits = self.audio_tagging_proj(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens) # (N,T)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits) # (N, num_events)
        if return_logits:
            return logits
        
        at_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        return at_loss
    
    def forward_speaker_verification(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        target: torch.Tensor = None,
        return_embedding: bool = False,
        loss_mode: str="cosine",
    ):
        # if loss_mode=cosine, target should be ()
        encoder_out = encoder_out.permute(0,2,1)
        encoder_out_lens = encoder_out_lens / torch.max(encoder_out_lens)
        asp_embedding = self.asp(encoder_out, encoder_out_lens) # (N,C,T)
        asp_embedding = asp_embedding.permute(0,2,1)
        asp_embedding = self.speaker_proj(asp_embedding) # (N, 1, 192)
        
        if return_embedding:
            return asp_embedding
        
        if loss_mode == "cosine":
            loss = 1 - F.cosine_similarity(asp_embedding, target, dim=-1, eps=1e-6)
        else:
            logits = self.speaker_pred(asp_embedding.squeeze(dim=1))
            loss = F.cross_entropy(logits, target, ignore_index=-100, reduction="none")
        return loss
        
        
