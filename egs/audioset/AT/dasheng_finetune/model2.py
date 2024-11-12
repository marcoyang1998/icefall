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

import logging
import random

import torch
import torch.nn as nn

from icefall.utils import make_pad_mask


class AudioTaggingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 768,
        num_encoder_layers: int = 12,
        num_events: int = 527,
        weighted_combine: bool = False,
        layer_idx: int = -1,
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
        self.num_encoder_layers = num_encoder_layers
        assert num_encoder_layers == len(encoder.blocks)
        
        self.layer_weight = nn.Parameter(torch.full((num_encoder_layers+1,), 0.5))
        self.classifier = nn.Linear(encoder_dim, num_events)
        
        self.weighted_combine = weighted_combine
        self.layer_idx = layer_idx
        self.freeze_encoder = freeze_encoder
        
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
        
    def forward_encoder(
        self,
        x: torch.Tensor,
    ):
        # obtain the merged hidden states from all layers
        with torch.set_grad_enabled(not self.freeze_encoder):
            x, all_hidden_states = self.encoder(x, output_hidden_states=True)
        if self.weighted_combine:
            all_hidden_states = torch.stack(all_hidden_states, dim=0) # (L,B,T,C)
            weight = torch.nn.functional.softmax(self.layer_weight, dim=0) #(L)
            if random.random() < 0.05 and self.training:
                logging.info(f"Current weight for each layer: {weight.tolist()}")
            all_hidden_states = weight.reshape(-1, 1,1,1) * all_hidden_states # (L,B,T,C) 
            hidden_states = torch.sum(all_hidden_states, dim=0) # (B,T,C)
        else:
            hidden_states = all_hidden_states[self.layer_idx] # (B,T,C)
        
        return hidden_states
    
    def forward_audio_tagging(
        self, x: torch.Tensor,
    ):
        return self.classifier(x).mean(dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ):  
        x = self.forward_encoder(x)
        logits = self.forward_audio_tagging(x) 
        loss = self.criterion(logits, target)
        
        return loss

    