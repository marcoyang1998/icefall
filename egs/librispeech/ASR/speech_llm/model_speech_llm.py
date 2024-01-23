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

import whisper

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from model import MultiKDModel

from icefall.utils import add_sos, make_pad_mask, AttributeDict
from scaling import ScaledLinear, SwooshR

class SpeechLLMModel(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        speech_encoder: MultiKDModel,
        vocab_size: int,
        llm_embed_dim: int = 1536,
        speech_encoder_dim: int = 2560,
        do_avg_pooling: bool = False,
        pad_token: int = 0,
        llm_requires_bos_id: bool = False,
    ):
        super().__init__()
        self.speech_encoder = speech_encoder # a pre-trained speech encoder
        self.speech_encoder_dim = speech_encoder_dim
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        
        self.llm = llm # a pre-trained LLM
        
        if do_avg_pooling:
            self.pooling_layer = nn.AvgPool1d(2, stride=2)
        else:
            self.pooling_layer = None
        
        self.embed_projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(speech_encoder_dim, llm_embed_dim),
            SwooshR(), # use the swooshR non-lin
        )

        self.audio_sos_eos_embedding = nn.Embedding(2, llm_embed_dim)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        
    def forward_LLM(
        self,
        y: torch.Tensor,
        y_lens: torch.Tensor,
    ):
        """Forward the LLM with the augmented input sequence.
        This function only takes the embeddings as input

        Args:
            y (torch.Tensor): The input embeddings, which is prepended with speech embedding
            (B,T,C)
            y_lens (torch.Tensor): The length of the augmented input_embeddings
        """
        src_key_padding_mask = ~ make_pad_mask(y_lens, max_len=y.shape[1])
        
        self.llm.eval() # Set the LLM param to eval mode
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
        import pdb; pdb.set_trace()
        encoder_out, encoder_out_lens, _ = self.speech_encoder.forward_encoder(
            x, x_lens, return_middle_out=False,
        ) # (N,T,C)
        if self.speech_encoder.whisper_projection is not None:
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
        text_prompt_lens: torch.Tensor = 0,
    ):
        """Forward the SpeechLLM

        Args:
            x (torch.Tensor): The input filter bank features
            x_lens (torch.Tensor): The length of the input audio
            y (torch.Tensor): The text tokens (prompt text + target text)
            y_lens (torch.Tensor): The length of the text tokens
            text_prompt_lens (torch.Tensor, optional): The lens of the prompt text. Defaults to 0.
        """
        device = x.device
        x, x_lens = self.encode_audio(x, x_lens) # (N,T,C)
        x_lens += 2 # we will add an <soa> and an <eoa> embedding to the audio
        
        y_embed = self.llm.get_input_embeddings()(y) # (N,U,C)
        concatenated_tokens, total_lens = self.concat_token_embedings(x, x_lens, y_embed, y_lens) # (N,U,C))
        
        logits = self.forward_LLM(concatenated_tokens, total_lens)
        
        bs = logits.shape[0]
        
        # Construct the mask of tokens for loss computations, True on those positions
        mask = torch.arange(logits.shape[1]).expand(bs, -1).to(device)
        # The prediction at (x_lens -1)-th position should be the first non-speech token
        mask = mask >= (x_lens - 1 + text_prompt_lens).unsqueeze(-1) # the positions of all target text tokens should be True
        padding_mask = ~ make_pad_mask(total_lens - 1, max_len=total_lens.max()) # the positions on the padding token & eos token is False
        mask = torch.logical_and(mask, padding_mask)
        
        # The actual logits for loss computation
        kept_logits = logits[mask == True] # (N, C)
        
        # The actual target tokens (excluding prompt tokens) for loss computation
        label_mask = make_pad_mask(text_prompt_lens, max_len=y_lens.max()) 
        shift_labels = y[label_mask]
        kept_labels = shift_labels[shift_labels != self.pad_token] # throw away padding token
        
        loss = self.criterion(kept_logits, kept_labels)
        
        return loss
    
    def concat_token_embedings(self, x, x_lens, y, y_lens):
        # Concat the audio token embeddings with the text token embeddings
        # This also works with empty sequence of text tokens
        # The audio embeddings is pre-pended with soa_embedding and appended with eoa_embedding
        bs = x.shape[0]
        new_tensors = [] # List[Tensor], each (B,T+U,C)
        soa_embedding = self.audio_sos_eos_embedding.weight[0, None]
        eoa_embedding = self.audio_sos_eos_embedding.weight[1, None]
        for i in range(bs):
            new_tensor = torch.cat((soa_embedding, x[i, :x_lens[i]-2], eoa_embedding, y[i]), dim=0) # <soa> audio <eoa> text prompt
            new_tensors.append(new_tensor)
        new_tensors = nn.utils.rnn.pad_sequence(new_tensors, batch_first=True)
        total_lens = x_lens + y_lens
        
        new_tensors = new_tensors[:, :total_lens.max(), :] # truncate to the maximal length
        
        return new_tensors, total_lens

    def embed_tokens(self, y):
        return self.llm.get_input_embeddings()(y)

    def encode_audio(self, x, x_lens):
        x, x_lens = self.forward_speech_encoder(x, x_lens) # (N,T,C)
        x = self.embed_projection(x) # (N,T,C)

        return x, x_lens

    def generate(self, x, x_lens):
        x, x_lens = self.encode_audio(x, x_lens)
        
class WhisperEncoder(nn.Module):
    def __init__(
        self,
        whisper_version: str = "base.en",
    ):
        super(WhisperEncoder, self).__init__()
        whisper_model = whisper.load_model(whisper_version)
        
        import pdb; pdb.set_trace()
        self.encoder_dim = whisper_model.dims.n_audio_state
        self.model = whisper_model.encoder
        
        
    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        return_middle_out: bool = False,
        layer_idx: int = -1,
    ):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        x_lens: torch.Tensor, shape = (batch_size)
        layer_idx: which layer's output to use
        """
        x = F.gelu(self.model.conv1(x))
        x = F.gelu(self.model.conv2(x))
        x = x.permute(0, 2, 1)
        x_lens = torch.floor((x_lens + 1)/2).int()
        
        # make the model compatible with any input length
        mask = make_pad_mask(x_lens, max_len=1500).to(x.device)
        pos_emb = self.model.positional_embedding.masked_fill(mask.unsqueeze(-1), 0.0)
        x = (x + pos_emb[:,:x_lens.max(),:]).to(x.dtype)
        
        results = []
        for block in self.model.blocks:
            x = block(x)
            results.append(x)
        if layer_idx == -1: # use the last layer
            x = self.model.ln_post(x) # (N,T,C)
        else:
            x = results[layer_idx] # zero-based index

        return x, x_lens