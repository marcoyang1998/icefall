# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from wav2vec2_utils import Fp32GroupNorm, Fp32LayerNorm, TransposeLast


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        output_dim: int = 256,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
            
        conv_output_dim = conv_layers[-1][0]
        self.output_proj = nn.Linear(conv_output_dim, output_dim)
        self.downsample_rate = np.prod([cl[2] for cl in conv_layers])
        
        # Store conv_layers config for output length computation
        self.conv_layers_config = conv_layers

    def get_downsample_rate(self):
        return self.downsample_rate
    
    def get_output_lengths(self, input_lengths):
        """
        Compute the output length after passing through all conv layers.
        
        Args:
            input_lengths: torch.Tensor of shape (N,) containing input lengths
            
        Returns:
            torch.Tensor of shape (N,) containing output lengths
        """
        lengths = input_lengths.clone()
        
        for i, (dim, k, stride) in enumerate(self.conv_layers_config):
            # For Conv1d with no padding and dilation=1:
            # output_length = floor((input_length - kernel_size) / stride) + 1
            lengths = torch.div(lengths - k, stride, rounding_mode='floor') + 1
            
        return lengths
    
    def forward(self, x, x_lens = None):
        # BxT -> BxTxC
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)
        x = self.output_proj(x)
        
        if x_lens is not None:
            x_lens = self.get_output_lengths(x_lens)

        return x, x_lens