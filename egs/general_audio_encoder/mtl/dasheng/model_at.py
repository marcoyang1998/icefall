# Copyright    2025  University of Cambridge        (authors: Xiaoyu Yang,
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

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from icefall.utils import make_pad_mask

 
class MultiTaskModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 384,
        num_events: int = 527,
        layer_idx: int = -1,
        post_norm: bool = False
    ):
        """A audio tagging model

        Args:
          encoder:
            Dasheng Encoder
          num_events:
            How man events
        """
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.layer_idx = layer_idx
        
        if post_norm:
            self.norm = nn.LayerNorm(encoder_dim)
        else:
            self.norm = None
        
        self.audio_tagging_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        ) # 527 classes

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, freeze_encoder: bool=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 2-D tensor of shape (N, T).
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
        with torch.set_grad_enabled((not freeze_encoder) and self.training):
            encoder_out, encoder_out_lens = self.encoder.get_embeddings(
                x, x_lens ,layer_idx=self.layer_idx,
            )
            if encoder_out_lens.max() < encoder_out.shape[1]:
                encoder_out = encoder_out[:, :encoder_out_lens.max(), :]
        if self.norm is not None:
            encoder_out = self.norm(encoder_out)

        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        at_targets: torch.Tensor,
        freeze_encoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, T), audio waveform.
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
        Returns:
          Return the audio tagging loss
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(
            x, 
            x_lens, 
            freeze_encoder=freeze_encoder
        )
        
        loss = self.forward_audio_tagging(
            encoder_out,
            encoder_out_lens,
            at_targets,
            return_logits=False
        )
        
        return loss
    
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