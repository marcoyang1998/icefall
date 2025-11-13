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

from typing import Dict

from dasheng import dasheng_base, dasheng_06B, dasheng_12B
from dasheng.pretrained.pretrained import Dasheng
import torch
import torch.nn as nn

PRETRAINED_CHECKPOINTS = {
    'dasheng_base':
    'download/models/dasheng_base.pt',
    'dasheng_06B':
    'download/models/dasheng_06b.pt',
    'dasheng_12B':
    'download/models/dasheng_12b.pt',
}


def dasheng_base(**model_kwargs):
    model_kwargs["embed_dim"] = 768
    model_kwargs["depth"] = 12
    model_kwargs["num_heads"] = 12
    return Dasheng.from_pretrained(PRETRAINED_CHECKPOINTS['dasheng_base'],
                                   **model_kwargs)


def dasheng_06B(**model_kwargs):
    model_kwargs["embed_dim"] = 1280
    model_kwargs["depth"] = 32
    model_kwargs["num_heads"] = 16
    return Dasheng.from_pretrained(PRETRAINED_CHECKPOINTS['dasheng_06B'],
                                   **model_kwargs)


def dasheng_12B(**model_kwargs):
    model_kwargs["embed_dim"] = 1536
    model_kwargs["depth"] = 40
    model_kwargs["num_heads"] = 24
    return Dasheng.from_pretrained(PRETRAINED_CHECKPOINTS['dasheng_12B'],
                                   **model_kwargs)


def get_encoder_model(model_version) -> nn.Module:
    MODEL_DICT = {
        "base": dasheng_base,
        "medium": dasheng_06B,
        "large": dasheng_12B
    }
    encoder = MODEL_DICT[model_version]()
    return encoder

class DashengEncoder(nn.Module):
    def __init__(self, model_version: str) -> None:
        super().__init__()
        self.model = get_encoder_model(model_version)
        
    def get_embeddings(self, audio, audio_lens, layer_idx=-1):
        try:
            x = self.model(audio, layer_idx=layer_idx) # (B,T,C)
        except TypeError:
            x = self.model(audio)
        x_lens = (audio_lens / 16000 * 25).int() # the frame rate is 25 Hz
        
        return x, x_lens
    
    def extract_features(self, batch: Dict, layer_idx: int = -1):
        device = next(self.model.parameters()).device
        
        audio = batch["audio"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        
        return self.get_embeddings(audio, audio_lens, layer_idx)
        
        

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
        layer_idx: int = -1
    ):
        if layer_idx == -1:
            x, _ = self.encoder(x)
        else:
            _, all_hidden_states = self.encoder(x, output_hidden_states=True)
            x = all_hidden_states[layer_idx] # (B,T,C)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        layer_idx: int = -1,
    ):
        # layer_idx means which layer's feature to use for classification
        with torch.set_grad_enabled(not self.freeze_encoder):
            x = self.forward_encoder(x, layer_idx=layer_idx)
            
        logits = self.classifier(x) # (B,T,C)
        logits = logits.mean(dim=1)
        
        loss = self.criterion(logits, target)
        
        return loss

    