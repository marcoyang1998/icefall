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

import logging
import random
from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn

from icefall.utils import add_sos, make_pad_mask, time_warp


class AsrModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 1024,
        weighted_combine: bool = False,
        layer_idx: int = -1,
        num_encoder_layers: int = 24,
        freeze_encoder: bool = True,
        vocab_size: int = 500,
    ):
        """A simplest CTC model

        Args:
          
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
          use_attention_decoder:
            Whether use attention-decoder head. Default: False.
        """
        super().__init__()

        self.encoder = encoder
        self.num_encoder_layers = num_encoder_layers
        
        self.layer_weight = nn.Parameter(torch.full((num_encoder_layers+1,), 0.5))
        # Modules for CTC head
        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )
        
        self.weighted_combine = weighted_combine
        self.layer_idx = layer_idx
        self.freeze_encoder = freeze_encoder

    def forward_encoder(
        self, batch
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

        with torch.set_grad_enabled(not self.freeze_encoder):
            features, all_hidden_states, embedding_lens = self.encoder(
                batch, 
            ) # first is the final hidden state
        
        if self.weighted_combine:
            all_hidden_states = torch.stack(all_hidden_states, dim=0) # (L,B,T,C)
            weight = torch.nn.functional.softmax(self.layer_weight, dim=0) #(L)
            if random.random() < 0.05 and self.training:
                logging.info(f"Current weight for each layer: {weight.tolist()}")
            all_hidden_states = weight.reshape(-1, 1,1,1) * all_hidden_states # (L,B,T,C) 
            hidden_states = torch.sum(all_hidden_states, dim=0) # (B,T,C)
        else:
            if self.layer_idx == -1:
                hidden_states = features
            else:
                hidden_states = all_hidden_states[self.layer_idx] # (B,T,C)
            
        return hidden_states, embedding_lens
        

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

    def forward(
        self,
        batch,
        y: k2.RaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (N, T).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.

        Returns:
          Return the CTC loss

        """
        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(batch)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]
        
        # Compute CTC loss
        targets = y.values
        ctc_loss = self.forward_ctc(
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            targets=targets,
            target_lengths=y_lens,
        )

        return ctc_loss
