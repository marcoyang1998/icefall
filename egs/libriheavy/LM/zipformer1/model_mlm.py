#!/usr/bin/env python3
# Copyright    2023       Xiaomi Corp.        (authors: Daniel Povey)
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

import random

import torch
from torch import nn, Tensor
from subformer import Subformer
from scaling import Balancer

from icefall.utils import make_pad_mask


class TextEmbedder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim,
                               embedding_dim,
                               groups=embedding_dim,
                               kernel_size=2)
        self.balancer1 = Balancer(embedding_dim,
                                  channel_dim=1,
                                  min_positive=0.1,
                                  min_abs=1.0,
                                  max_abs=2.0)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv1d(embedding_dim,
                               embedding_dim,
                               kernel_size=2)

        self.balancer2 = Balancer(embedding_dim,
                                  channel_dim=1,
                                  min_positive=0.1,
                                  min_abs=1.0,
                                  max_abs=2.0)
        self.activation2 = nn.ReLU()

        self.out_proj = nn.Linear(embedding_dim,
                                  embedding_dim,
                                  bias=False)

    def forward(self,
                text: Tensor) -> Tensor:
        """
        Args:
            text: Tensor of shape (seq_len, batch_size), containing integer indexes
                 0 <= text < vocab_size.
        Returns:
            Tensor of shape (seq_len, batch_size, embedding_dim)
        """
        x = self.embed(text)  # (seq_len, batch_size, embedding_dim)

        x = x.permute(1, 2, 0)  # N,C,H, i.e. (batch_size, embedding_dim, seq_len)
        x = torch.nn.functional.pad(x, (1, 0))
        x = self.conv1(x)
        x = self.balancer1(x)  # make sure no channel has all zeros.
        x = self.activation1(x)
        x = torch.nn.functional.pad(x, (1, 0))
        x = self.conv2(x)
        x = self.balancer2(x)
        x = self.activation2(x)
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, embedding_dim)
        x = self.out_proj(x)
        return x

class SubformerLM(nn.Module):

    def __init__(self,
                 encoder_embed: nn.Module,
                 encoder: Subformer,
                 decoder: nn.Module,
                 vocab_size: int=256,
    ):
        # Note, this is a subformer LM model for BERT-like masked language modelling
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder # does subsampling
        self.decoder = decoder
        self.vocab_size = vocab_size

    def _create_masked_input_and_labels(
        self,
        text: Tensor,
        text_lens: Tensor
    ):
        device = text.device
        src = text
        
        batch_size, max_len = text.size()
        lens_mask = torch.arange(max_len).expand(batch_size, max_len).to(device) < text_lens.unsqueeze(-1).to(device)
        # mask 15% tokens for MLM
        mask = torch.rand(text.size(1), device=device) <= 0.15  # (N,T)
        mask = mask.expand(text.shape)
        mask = mask * lens_mask.to(device)
        label = text[mask] # (N,T')
        text[mask] = 0 # fill with byte 0
        v = random.random()
        
        # 10% of the masked tokens use a random token
        mask2 = torch.rand(text.size(1), device=device) <=0.1 
        mask2 = mask * mask2
        mask2 = mask2.expand(text.shape)
        random_tokens = torch.randint_like(text, low=1, high=255)
        text[mask2] = random_tokens[mask2]
        
        # 10% of the masked tokens use the original token
        mask3 = torch.rand(text.size(1), device=device) <=0.1
        mask3 = mask * mask3
        mask3 = mask3.expand(text.shape)
        text[mask3] = src[mask3]
        
        return text, mask, label

    def forward(self,
            text: Tensor,
            text_lens: Tensor
        ):
        """
        Compute array of log-probs

        Args:
          text: a Tensor containing the labels (in the range 0..num_symbols-1), of shape (batch_size, seq_len).
          mask: a Tensor masking the input sequence, where we compute the MLM loss
        Returns:
           a Tensor containing the log-probs for each label, of shape (batch_size, seq_len).
        """
        device = text.device
        (batch_size, seq_len) = text.shape
        orig_labels = text.clone()
        
        text, mask, masked_labels = self._create_masked_input_and_labels(text, text_lens)
        text_key_padding_mask = make_pad_mask(text_lens).to(device)
        
        chunk_size = 1
        text_shifted = text.t()  # (time, batch)
        text_shifted = torch.cat((torch.zeros_like(text_shifted[:1]),
                                    text_shifted[:-1]),
                                   dim=0)

        x = self.encoder_embed(text_shifted)
        x_lens = torch.full((batch_size,), seq_len,
                            dtype=torch.long, device=device)

        # x_lens is after subsampling.  Actually we don't need it.
        (x, x_lens) = self.encoder(x, x_lens)

        logits = self.decoder(orig_labels, x, return_logits=True) # (batch_size, seq_len, vocab_size)
        masked_logits = logits[mask] # (bs, seq_len', vocab_size)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fn(masked_logits, masked_labels)
        
        return loss
