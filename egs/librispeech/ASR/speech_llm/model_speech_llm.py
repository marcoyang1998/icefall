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
        vocab_size: int,
        llm_embed_dim: int = 1536,
        speech_encoder_dim: int = 2560,
        do_avg_pooling: bool = False,
    ):
        super().__init__()
        self.speech_encoder = speech_encoder # a pre-trained speech encoder
        self.speech_encoder_dim = speech_encoder_dim
        self.vocab_size = vocab_size
        
        self.llm = llm # a pre-trained LLM

        # set requires_grad=False for all the parameters in llm
        for param in self.llm.parameters():
            param.requires_grad = False
        
        if do_avg_pooling:
            self.pooling_layer = nn.AvgPool1d(2, stride=2)
        else:
            self.pooling_layer = None
        
        self.embed_projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(speech_encoder_dim, llm_embed_dim),
            nn.Tanh(),
        )
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        
    def forward_LLM(
        self,
        y: torch.Tensor,
        y_lens: torch.Tensor,
    ):
        """Forward the LLM with the augmented input sequence

        Args:
            y (torch.Tensor): The input embeddings, which is prepended with speech embedding
            (B,T,C)
            y_lens (torch.Tensor): The length of the augmented input_embeddings
        """
        import pdb; pdb.set_trace()
        src_key_padding_mask = ~ make_pad_mask(y_lens, max_len=y.shape[1])
        
        output = self.llm(
            inputs_embeds=y,
            attention_mask=src_key_padding_mask,
            use_cache=False
        ).logits
        return output
    
    def forward_speech_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ):
        """Get the output of the speech encoder

        Args:
            x (torch.Tensor): The input filterbank features, (N,T,C)
            x_lens (torch.Tensor): The input length

        """
        encoder_out, encoder_out_lens, _ = self.speech_encoder.forward_encoder(
            x, x_lens, return_middle_out=False,
        ) # (N,T,C)
        if hasattr(self.speech_encoder, 'whisper_projection'):
            encoder_out = self.speech_encoder.whisper_projection(encoder_out)
            
        if self.pooling_layer is not None:
            encoder_out = encoder_out.permute(0,2,1) # (N,C,T), required by pooling
            encoder_out = self.pooling_layer(encoder_out) 
            encoder_out = encoder_out.permute(0,2,1) # (N,T,C)
            encoder_out_lens = encoder_out_lens // 2
        
        return encoder_out, encoder_out_lens
    
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
    ):
        import pdb; pdb.set_trace()
        x, x_lens = self.forward_speech_encoder(x, x_lens) # (N,T,C)
        x = self.embed_projection(x) # (N,T,C)
        
        import pdb; pdb.set_trace()
        y_embed = self.llm.get_input_embeddings()(y) # (N,U,C)
        concatenated_tokens, total_lens = self.concat_token_embedings(x, x_lens, y_embed, y_lens) # (N,U,C))
        
        import pdb; pdb.set_trace()
        logits = self.forward_LLM(concatenated_tokens, total_lens)
        
        bs = logits.shape[0]
        shift_logits = logits[...,:-1,:].contiguous() # We don't compute loss after EOS
        
        # Construct the mask of textual tokens, True on those positions
        # +2 because we don't want to compute loss on the tag token
        mask = torch.arange(logits.shape[1]).expand(bs, -1)
        mask = mask > (x_lens + 2).unsqueeze(-1) # the positions of text token is True
        padding_mask = ~ make_pad_mask(total_lens -1) # the positions on the padding token is False
        mask = torch.logical_and(mask, padding_mask)
        
        # The actual logits for loss computation
        kept_logits = shift_logits[mask == True] # (N, C)
        
        # The actual labels for loss computation
        shift_labels = y[:, 2:].contiguous() # throw away <bos> token
        kept_labels = shift_labels[shift_labels > 0] # throw away padding token
        
        loss = self.criterion(kept_logits, kept_labels)
        
        return loss
    
    @staticmethod
    def concat_token_embedings(x, x_lens, y, y_lens):
        bs = x.shape[0]
        new_tensors = [] # List[Tensor], each (B,T+U,C)
        import pdb; pdb.set_trace()
        for i in range(bs):
            new_tensor = torch.cat((x[i, :x_lens[i]], y[i]), dim=0)
            new_tensors.append(new_tensor)
        new_tensors = nn.utils.rnn.pad_sequence(new_tensors, batch_first=True)
        total_lens = x_lens + y_lens
        
        import pdb; pdb.set_trace()
        new_tensors = new_tensors[:, :total_lens.max(), :] # truncate to the maximal length
        
        return new_tensors, total_lens
        
