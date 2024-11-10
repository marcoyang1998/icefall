# Copyright    2021-2023  University of Cambridge     (authors: Xiaoyu Yang,
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

from icefall.utils import make_pad_mask


class AudioTaggingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        num_events: int = 527,
        freeze_encoder: bool = True,
    ):
        """The simplest audio tagging model. 

        Args:
            encoder (nn.Module): The encoder that processes the input audio
            encoder_dim (int): The dimension of the encoder
            num_events (int, optional): The number of audio events. Defaults to 527.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.classifier = nn.Linear(encoder_dim, num_events)
        self.freeze_encoder = freeze_encoder
        
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
        
    def forward_encoder(
        self,
        x: torch.Tensor,
    ):
        x, _ = self.encoder(x)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ):
        with torch.set_grad_enabled(not self.freeze_encoder):
            x = self.forward_encoder(x)
            
        logits = self.classifier(x) # (B,T,C)
        logits = logits.mean(dim=1)
        
        loss = self.criterion(logits, target)
        
        return loss

    