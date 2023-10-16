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

from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface

from icefall.utils import add_sos, make_pad_mask
from scaling import ScaledLinear

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling


class MultiKDModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        encoder_dim: int = 384,
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
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        
        self.beats_decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, 527),
        ) # 527 classes
        self.ecapa_asp = AttentiveStatisticsPooling(channels=encoder_dim)
        self.ecapa_linear = nn.Linear(2 * encoder_dim, 192 )# fixed 192-D vector

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
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

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        teacher_beats_embeddings: torch.Tensor = None,
        teacher_ecapa_embeddings: torch.Tensor = None,
        teacher_whisper_embeddings: torch.Tensor = None,
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
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens) # (N,T,C)

        # beats loss
        if teacher_beats_embeddings is not None:
            beats_logits = self.forward_beats(encoder_out, encoder_out_lens)
            
            # normalize the teacher probabilities
            teacher_beats_embeddings = teacher_beats_embeddings / teacher_beats_embeddings.sum(dim=-1).unsqueeze(-1).expand_as(teacher_beats_embeddings)
            
            beats_loss = F.kl_div(beats_logits, teacher_beats_embeddings, reduction="sum")
        else:
            beats_loss = torch.empty(0)
        
        # ecapa loss
        if teacher_ecapa_embeddings is not None:
            encoder_out = encoder_out.permute(0,2,1)
            ecapa_embeddings = self.ecapa_asp(encoder_out) # (N,C,T)
            ecapa_embeddings = ecapa_embeddings.permute(0,2,1)
            ecapa_embeddings = self.ecapa_linear(ecapa_embeddings) # (N, 1, 192)
            ecapa_loss = 1 - F.cosine_similarity(ecapa_embeddings, teacher_ecapa_embeddings, dim=-1, eps=1e-6)
            ecapa_loss = ecapa_loss.sum()
        else:
            ecapa_loss = torch.empty(0)
        
        whisper_loss = torch.empty(0)
        
        return beats_loss, ecapa_loss, whisper_loss
    
    def forward_beats(
        self,
		encoder_out: torch.Tensor,
  		encoder_out_lens: torch.Tensor,
	):
        beats_logits = self.beats_decoder(encoder_out) # (N,)
        padding_mask = make_pad_mask(encoder_out_lens) 
        beats_logits[padding_mask] = 0
        beats_logits = beats_logits.sum(dim=1)
        beats_logits = beats_logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(beats_logits)
        beats_logits = torch.log_softmax(beats_logits, dim=-1).unsqueeze(1)
        
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