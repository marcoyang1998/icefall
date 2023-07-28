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
                 decoder: nn.Module):
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder # does subsampling
        self.decoder = decoder


    def forward(self,
                labels: Tensor):
        """
        Compute array of log-probs

        Args:
         labels: a Tensor containing the labels (in the range 0..num_symbols-1), of shape (batch_size, seq_len).
        Returns:
           a Tensor containing the log-probs for each label, of shape (batch_size, seq_len).
        """
        (batch_size, seq_len) = labels.shape

        x_lens = (labels > 0).sum(-1).long() # because we prepend sos 
        labels = labels[:, :x_lens.max()]
        chunk_size = 1
        labels_shifted = labels.t()  # (time, batch)
        labels_shifted = torch.cat((torch.zeros_like(labels_shifted[:1]),
                                    labels_shifted[:-1]),
                                   dim=0)

        x = self.encoder_embed(labels_shifted)

        # x_lens is after subsampling.  Actually we don't need it.
        src_key_padding_mask = make_pad_mask(x_lens)
        (x, x_lens) = self.encoder(x, x_lens, src_key_padding_mask)

        logprobs = self.decoder(labels, x)
        assert not logprobs.isnan().any()
        logprobs.masked_fill_(src_key_padding_mask, 0) # mask 
        
        return logprobs
