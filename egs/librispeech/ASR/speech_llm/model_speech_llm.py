# Copyright    2021-2023  Xiaomi Corp.        (authors: Xiaoyu)
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
from model import MultiKDModel

from icefall.utils import add_sos, make_pad_mask, AttributeDict
from scaling import ScaledLinear

class SpeechLLMModel(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        speech_encoder: MultiKDModel,
        llm_embed_dim: int = 1536,
        speech_encoder_dim: int = 2560,
    ):
        super().__init__()
        self.speech_encoder = speech_encoder # a pre-trained speech encoder
        self.speech_encoder_dim = speech_encoder_dim
        
        self.llm = llm # a pre-trained LLM
        
        self.embed_projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(speech_encoder_dim, llm_embed_dim),
            nn.ReLU(),
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward_LLM(
        self,
        y: torch.Tensor,
        y_lens: nn.Tensor,
    ):
        pass
    
    def forward_speech_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ):
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
    ):
        x, x_lens = self.forward_speech_encoder(x, x_lens) # (N,T,C)
        x = self.embed_projection(x) # (N,T,C)
        y, y_lens = self.forward_LLM(y, y_lens)
        
        loss = self.criterion()
        
        return
    
    @staticmethod
    def concat_token_embedings(x, x_lens, y, y_lens):
        pass
        
