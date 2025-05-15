# Copyright    2025 University of Cambridge      (authors: Xiaoyu Yang)
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

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from multi_quantization.prediction import JointCodebookLoss
from zipformer2 import Zipformer2
from scaling import FloatLike, ScheduledFloat

from icefall.utils import make_pad_mask


class MvqDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 768,
        num_codebooks: int = 16,
        frame_ratio: int = 2,
        delta: int = 0,
        interpolate_teacher: bool = False,
        output_downsampling_factor: int = 1,
        downsampling_factor: Tuple[int] = (2, 4),
        decoder_dim: Union[int, Tuple[int]] = 384,
        num_decoder_layers: Union[int, Tuple[int]] = 4,
        decoder_unmasked_dim: Union[int, Tuple[int]] = 256,
        query_head_dim: Union[int, Tuple[int]] = 24,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_dim: Union[int, Tuple[int]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = None,  # see code below for default
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size: Tuple[int] = [-1],
        left_context_frames: Tuple[int] = [-1],
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        decoder_input_dim = decoder_dim[0] if isinstance(decoder_dim, Tuple) else decoder_dim
        decoder_output_dim = max(decoder_dim) if isinstance(decoder_dim, Tuple) else decoder_dim
        
        self.in_proj = nn.Linear(encoder_dim, decoder_input_dim)
        
        # the actual decoder
        self.decoder = Zipformer2(
            output_downsampling_factor=output_downsampling_factor,
            downsampling_factor=downsampling_factor,
            num_encoder_layers=num_decoder_layers,
            encoder_dim=decoder_dim,
            encoder_unmasked_dim=decoder_unmasked_dim,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            cnn_module_kernel=cnn_module_kernel,
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            warmup_batches=4000.0,
            causal=False,
        )
        
        self.frame_ratio = frame_ratio
        self.num_codebooks = num_codebooks
        self.distillation_delta = delta
        self.interpolate_teacher = interpolate_teacher
                
        # cb index prediction
        self.predictor = JointCodebookLoss(
            predictor_channels=decoder_output_dim,
            num_codebooks=num_codebooks * frame_ratio,
            is_joint=False,
            reduction="none",
        )
        
    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, target: torch.Tensor
    ):
        """Forward MVQ decoder and compute MVQ loss

        Args:
            x (torch.Tensor): The encoder output, (B,T,C)
            x_lens (torch.Tensor): The length of each encoder output, (B,)
            target (torch.Tensor): The target MVQ tokens (B,T,num_cb)
        """
        x = self.in_proj(x)
        x, x_lens = self.decoder(x, x_lens)
        cb_loss = self.forward_codebook_loss(
            encoder_out=x, encoder_out_lens=x_lens, codebook_indexes=target
        )
        return cb_loss
        
        
    def forward_codebook_loss(
        self, 
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        codebook_indexes: torch.Tensor,
    ):
        """Compute the codebook loss, potentially concatenate or interpolate the target

        Args:
            encoder_out (torch.Tensor): The input to the prediction module
            encoder_out_lens (torch.Tensor): The lengths of each sample
            codebook_indexes (torch.Tensor): The target cb indexes

        """
        # align the encoder features with the codebook indexes
        if self.interpolate_teacher:
            codebook_indexes = self.interpolate_codebook_indexes(
                encoder_out, codebook_indexes
            )
        else:
            if codebook_indexes.shape[1] != encoder_out.shape[1]:
                # align the codebook indexes to the frame rate of the student encoder out
                codebook_indexes = self.concat_successive_codebook_indexes(
                    encoder_out, codebook_indexes, ratio=self.frame_ratio
                )
                
        # the delta is associated with the frame-rate of the student encoder
        # so a bigger delta maybe necessary for 50Hz student encoder
        if self.distillation_delta > 0:
            codebook_indexes = codebook_indexes[:,:-self.distillation_delta, :]
            encoder_out = encoder_out[:, self.distillation_delta:, :]
            truncated_padding_mask = make_pad_mask(encoder_out_lens - self.distillation_delta)
            codebook_indexes = codebook_indexes.masked_fill(truncated_padding_mask.unsqueeze(-1), value=-100)
            
        # compute the loss
        N,T,_ = encoder_out.shape
        codebook_loss = self.predictor(encoder_out.float(), codebook_indexes)
        codebook_loss = codebook_loss.reshape(N,T,-1)
        num_cb = codebook_loss.size(-1) # this is the equivalent number of codebooks
        # normalize the loss by the number of codebooks
        codebook_loss = codebook_loss.sum(dim=(1,2)) / num_cb
        
        return codebook_loss
    
    @staticmethod
    def interpolate_codebook_indexes(middle_layer_output, codebook_indexes):
        # This function addresses the case where the teacher has a lower frame rate
        # than the student model
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape # C should be 256
        
        codebook_indexes = codebook_indexes.permute(0,2,1).float() # (N,C,T)
        codebook_indexes = torch.nn.functional.interpolate(codebook_indexes, t_expected)
        codebook_indexes = codebook_indexes.permute(0,2,1).int() # (N,T,C)
        
        assert codebook_indexes.shape[1] == middle_layer_output.shape[1]
        return codebook_indexes
    
    @staticmethod
    def concat_successive_codebook_indexes(middle_layer_output, codebook_indexes, ratio=2):
        # Output rate of hubert is 50 frames per second,
        # while that of current encoder is 25.
        # Following code handling two issues:
        # 1.
        #   Roughly speaking, to generate another frame output,
        #   hubert needes extra two frames,
        #   while current encoder needs extra four frames.
        #   Suppose there are only extra three frames provided,
        #   hubert will generate another frame while current encoder does nothing.
        # 2.
        #   codebook loss is a frame-wise loss, to enalbe 25 frames studnet output
        #   learns from 50 frames teacher output, two successive frames of teacher model
        #   output is concatenated together.
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape # C should be 256
        
        # Handling issue 1.
        if T >= t_expected * ratio:
            codebook_indexes = codebook_indexes[:, : t_expected * ratio, :]
        else:
            assert t_expected * ratio - T <= 5, (T, t_expected, ratio)
            diff = t_expected * ratio - T
            codebook_indexes = torch.cat(
                [
                    codebook_indexes,
                    torch.full((N,diff,C), -100).to(codebook_indexes.device).to(codebook_indexes.dtype)
                ]
            )
        assert codebook_indexes.size(1) == middle_layer_output.size(1) * ratio
        
        # Handling issue 2.
        codebook_indexes = codebook_indexes.reshape(N, t_expected, C * ratio)
        assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
        return codebook_indexes
        
def _test_mvq_decoder():
    batch_size = 5
    seq_len = 20
    num_cb = 16
    frame_ratio = 2
    delta = 0
    encoder_dim=256
    
    model = MvqDecoder(
        encoder_dim=encoder_dim,
        frame_ratio=frame_ratio,
        delta=delta,
        num_codebooks=num_cb,
        output_downsampling_factor=1,
        num_decoder_layers=(2,2),
        downsampling_factor=(2, 2),
        decoder_dim=(128, 128),
        decoder_unmasked_dim=(48, 64),
        num_heads=(4, 4),
    )
    num_params = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters: {num_params}")
    x = torch.randn(batch_size, seq_len, encoder_dim)
    x_lens = torch.randint(15, seq_len, (batch_size, ))
    target = torch.randint(0, 255, (batch_size, seq_len * frame_ratio, num_cb))
    
    mvq_loss = model(x, x_lens, target)
    logging.info(mvq_loss.shape)
    
def _test_ce_loss():
    criteriorn = nn.CrossEntropyLoss(reduction="none")
    x = torch.ones(1,5)
    y = torch.randint(1, 5, (1,))
    
    my_loss = criteriorn(x, y)
    print(my_loss)
    
        
if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    # _test_ce_loss()
    _test_mvq_decoder()