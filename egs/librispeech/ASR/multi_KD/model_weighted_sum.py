# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
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

from typing import Optional, Tuple, List
import logging
import random

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface

from icefall.utils import add_sos, make_pad_mask, AttributeDict
from scaling import ScaledLinear

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling


class MultiKDModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        encoder_dim: int = 384,
        whisper_dim: int = 768,
        use_beats: bool = True,
        use_ecapa: bool = True,
        use_whisper: bool = True,
        speaker_input_idx: int = -1,
        mvq_KD: bool = False,
        num_codebooks: int = 32,
        mvq_kd_layer: int = -1,
        cb_input_dim: int = 512,
        use_subsampled_output: bool = True,
        delta_t: int = 0,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
          use_whisper:
            If use l1 embedding loss for whisper features
          mvq_KD:
            If use codebook loss for whisper teacher
          mvq_kd_layer:
            Use which zipformer block's output for MVQ-KD. If -1, use `encoder_out`, 
            Otherwise use middle_out[mvq_kd_layer][-1]
          speaker_input_idx:
            Which modules results to be inputted to the ecapa module
          use_subsampled_output:
            If use the final subsampled output i.e 25 Hz
          delta_t:
            If use a delta when computing the distillation loss for whisper
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.encoder_dims = self.encoder.encoder_dim # tuple
        self.num_layers = self.encoder.num_encoder_layers # tuple
        
        # KD modules
        self.use_beats = use_beats
        if use_beats:
            self.beats_decoder = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(encoder_dim, 527),
            ) # 527 classes
        else:
            self.beats_decoder = None
        
        self.use_ecapa = use_ecapa
        self.speaker_input_idx = speaker_input_idx
        if use_ecapa:
            if speaker_input_idx == -1: # use the last layer or the weighted sum
                self.ecapa_asp = AttentiveStatisticsPooling(channels=encoder_dim)
                self.ecapa_linear = nn.Linear(2 * encoder_dim, 192 ) # fixed 192-D vector
            else:
                speaker_input_dim = self.encoder_dims[speaker_input_idx]
                self.ecapa_asp = AttentiveStatisticsPooling(channels=speaker_input_dim)
                self.ecapa_linear = nn.Linear(2 * speaker_input_dim, 192 ) # fixed 192-D vector
        else:
            self.ecapa_asp = None
            self.ecapa_linear = None
            
        self.use_whisper = use_whisper
        if use_whisper:
            if use_subsampled_output:
                self.whisper_projection = nn.Linear(encoder_dim, 2 * whisper_dim) # a linear transform
            else:
                self.whisper_projection = nn.Linear(256, whisper_dim) # a linear transform
            self.codebook_loss_net = None
        else:
            self.whisper_projection = None
        
        self.delta_t = delta_t
        if delta_t > 0:
            logging.info(f"Delta_t: {delta_t} when computing the distillation loss")

        # if use codebook loss
        self.mvq_KD = mvq_KD
        self.mvq_layer = mvq_kd_layer
        self.codebook_loss_net = None


    def forward_encoder(
        self, 
        x: torch.Tensor,
        x_lens: torch.Tensor,
        return_middle_out: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens, middle_out = self.encoder(x, x_lens, src_key_padding_mask, return_middle_out=return_middle_out)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens, middle_out

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor = None,
        teacher_beats_embeddings: torch.Tensor = None,
        teacher_ecapa_embeddings: torch.Tensor = None,
        teacher_whisper_embeddings: torch.Tensor = None,
        teacher_whisper_embedding_lens: torch.Tensor = None,
        teacher_whisper_codebook_indexes: torch.Tensor = None,
        teacher_whisper_codebook_indexes_lens: torch.Tensor = None,
        return_middle_out: bool = False,
        reduction: str = "sum"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          teacher_beats_embeddings:
            BEATs teacher's embedding (N, 1, 527)
          teacher_ecapa_embeddings:
            Ecapa teacher's embedding (N, 1, 192)
          teacher_whisper_embeddings:
            Whisper teachers embeddings (N,T,C)
          teacher_whisper_embedding_lens:
            The length of the whisper teacher's embedding (N,)
          teacher_whisper_codebook_indexes:
            The whisper teacher's embeddings encoded in codebook indexes (N,T, num_cb)
          teacher_whisper_codebook_indexes_lens:
            The length of the whisper teacher's codebook indexes (N,)

        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        # assert y.num_axes == 2, y.num_axes

        # assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens, middle_out = self.forward_encoder(x, x_lens, return_middle_out=True) # (N,T,C)

        # beats loss
        if self.use_beats and teacher_beats_embeddings is not None:
            beats_logits = self.forward_beats(encoder_out, encoder_out_lens)
            teacher_beats_embeddings = teacher_beats_embeddings.squeeze(dim=1) # (N, num_events)
            beats_loss = F.binary_cross_entropy_with_logits(beats_logits, teacher_beats_embeddings, reduction=reduction)
        else:
            beats_loss = None
        
        # ecapa loss
        if self.use_ecapa and teacher_ecapa_embeddings is not None:
            assert middle_out is not None
            if self.speaker_input_idx == -1:
                ecapa_embeddings = self.forward_ecapa(
                    encoder_out, # (N,T,C)
                    encoder_out_lens,
                )
            else:
                ecapa_input_embeddings = middle_out[self.speaker_input_idx] # a list of (T,N,C)
                # ecapa_input_embeddings = torch.mean(torch.stack(ecapa_input_embeddings), dim=0)
                ecapa_input_embeddings = sum(ecapa_input_embeddings) / len(ecapa_input_embeddings)
                ecapa_input_embeddings = ecapa_input_embeddings.permute(1,0,2)

                ecapa_embeddings = self.forward_ecapa(
                    ecapa_input_embeddings,
                    encoder_out_lens,
                )
            
            ecapa_loss = 1 - F.cosine_similarity(ecapa_embeddings, teacher_ecapa_embeddings, dim=-1, eps=1e-6)
            if reduction == "sum":
                ecapa_loss = ecapa_loss.sum()
        else:
            ecapa_loss = None
        
        if self.use_whisper and teacher_whisper_embeddings is not None:
            whisper_embeddings = self.forward_whisper(encoder_out, encoder_out_lens)
            teacher_whisper_embeddings = self.concat_successive_whisper_embeddings(
                whisper_embeddings,
                teacher_whisper_embeddings,
            )
            
            mask = make_pad_mask(encoder_out_lens)
            if self.delta_t > 0:
                whisper_embeddings = whisper_embeddings[:, self.delta_t:, :] 
                teacher_whisper_embeddings = teacher_whisper_embeddings[:, :-self.delta_t, :]
                mask = mask[:, self.delta_t:]
                
            whisper_loss = F.l1_loss(whisper_embeddings, teacher_whisper_embeddings, reduction="none")
            whisper_loss.masked_fill_(mask.unsqueeze(-1), 0.0)
            
            if reduction == "sum":
                whisper_loss = whisper_loss.sum() / teacher_whisper_embeddings.shape[-1]
            else:
                whisper_loss = whisper_loss / teacher_whisper_embeddings.shape[-1]
        else:
            whisper_loss = None

        if (self.mvq_KD and self.training) and teacher_whisper_codebook_indexes is not None:
            if self.mvq_layer == -1:
                middle_layer = encoder_out
            else:
                # middle_layer = self.encoder.encoders[self.mvq_layer].upsample(middle_out[self.mvq_layer][-1]).permute(1,0,2)
                middle_layer = middle_out[self.mvq_layer][-1].permute(1,0,2)
            whisper_cb_loss = self.forward_codebook(
                middle_layer_output=middle_layer,
                codebook_indexes=teacher_whisper_codebook_indexes,
            )
        else:
            whisper_cb_loss = None
        
        return beats_loss, ecapa_loss, whisper_loss, whisper_cb_loss
    
    def forward_beats(
        self,
        encoder_out: torch.Tensor,
  		encoder_out_lens: torch.Tensor,
	):
        beats_logits = self.beats_decoder(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens) 
        beats_logits[padding_mask] = 0
        beats_logits = beats_logits.sum(dim=1)
        beats_logits = beats_logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(beats_logits) # (N, num_events)
        # beats_logits = torch.log_softmax(beats_logits, dim=-1).unsqueeze(1)
        
        return beats_logits
    
    def forward_ecapa(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        encoder_out = encoder_out.permute(0,2,1)
        encoder_out_lens = encoder_out_lens / torch.max(encoder_out_lens)
        ecapa_embeddings = self.ecapa_asp(encoder_out, encoder_out_lens) # (N,C,T)
        ecapa_embeddings = ecapa_embeddings.permute(0,2,1)
        ecapa_embeddings = self.ecapa_linear(ecapa_embeddings) # (N, 1, 192)
        
        return ecapa_embeddings
    
    def forward_whisper(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        return self.whisper_projection(F.relu(encoder_out))

    def forward_codebook(
        self,
        middle_layer_output: torch.Tensor,
        codebook_indexes: torch.Tensor,
    ):
        """Calculate the codebook loss for the model (knowledge distillation)

        Args:
                middle_layer_output (torch.Tensor):
                        The embeddings extracted from the middle layer of the zipformer encoder
                codebook_indexes (torch.Tensor):
                        The encoded codebook indexes for knowledge distillation

        Returns:
                The codebook loss value
        """
        len_CI = codebook_indexes.size(1)
        len_mid_layer = middle_layer_output.size(1)
        ratio = round(len_CI / len_mid_layer)

        if ratio == 1:  # Having the same frame rate
            assert len_CI > len_mid_layer, (len_CI, len_mid_layer)
            codebook_indexes = codebook_indexes[:, :len_mid_layer, :]
            assert codebook_indexes.size(1) == middle_layer_output.size(1)
            codebook_loss = self.codebook_loss_net(
                middle_layer_output, codebook_indexes
            )
        elif ratio >= 2:
            codebook_indexes = self.concat_successive_codebook_indexes(
                middle_layer_output, codebook_indexes, ratio=ratio
            )
            codebook_loss = self.codebook_loss_net(
                middle_layer_output, codebook_indexes
            )

        return codebook_loss
    
    @staticmethod
    def concat_successive_whisper_embeddings(encoder_out, whisper_embeddings):
        t_expected = encoder_out.shape[1]
        N, T, C = whisper_embeddings.shape
        
        if T >= t_expected * 2:
            whisper_embeddings = whisper_embeddings[:, : t_expected * 2, :]
            whisper_embeddings = whisper_embeddings.reshape(N, t_expected, C * 2)
        else:
            whisper_embeddings = whisper_embeddings[:, : t_expected, :]

        # An ugly way to avoid shape mismatch of dummy embedding, this only happens if the whole batch is dummy embedding!
        if whisper_embeddings.shape[1] != encoder_out.shape[1]:
            return encoder_out

        assert whisper_embeddings.shape[1] == encoder_out.shape[1]
        assert whisper_embeddings.shape[2] == encoder_out.shape[2], f"{N}x{T}x{C} {whisper_embeddings.shape}, {encoder_out.shape}"

        return whisper_embeddings

    @staticmethod
    def concat_successive_codebook_indexes(middle_layer_output, codebook_indexes, ratio=2):
        # Output rate of hubert is 50 frames per second,
        # while that of current encoder is 25.
        # Following code handling two issues:
        # 1.
        #   Roughly speaking, to generate another frame output,
        #   hubert needes extra two frames,
        #   while current encoder needs extra four frames.
        #   Suppose there are only extra three frames provided,
        #   hubert will generate another frame while current encoder does nothing.
        # 2.
        #   codebook loss is a frame-wise loss, to enalbe 25 frames studnet output
        #   learns from 50 frames teacher output, two successive frames of teacher model
        #   output is concatenated together.
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape
        assert T >= t_expected, (T, t_expected)
        # Handling issue 1.
        if T >= t_expected * 2:
            codebook_indexes = codebook_indexes[:, : t_expected * ratio, :]
        if (
            T / t_expected < 1.1
        ):  # To be changed, dirty hack to jump out of this function
            codebook_indexes = codebook_indexes[:, :t_expected, :]
            assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
            return codebook_indexes
        # Handling issue 2.
        codebook_indexes = codebook_indexes.reshape(N, t_expected, C * ratio)
        assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
        return codebook_indexes
    