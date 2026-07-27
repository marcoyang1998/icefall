# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
#
# Copyright    2024 University of Cambridge      (authors: Xiaoyu Yang)
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
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_quantization.prediction import JointCodebookLoss

from icefall.utils import make_pad_mask

class SimpleDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(self, channels: int, downsample: int):
        super(SimpleDownsample, self).__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None  # will be set from training code

        self.downsample = downsample

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        # right-pad src, repeating the last element.
        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        # weights: (downsample, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)

        return ans


class MultiKDModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        num_codebooks: list[int]=None,
        distillation_layer: list[int]=None,
        distillation_delta: list[int]=None,
        teacher_frame_ratio: list[int]=None,
        interpolate_teacher: bool = False,
        output_downsampling_factor: int = 1,
        loss_only_mask: bool = False,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          num_codebooks:
            A list of integers, how many codebooks for each target
          mask_mode:
            The masking mode.
                w2v2: the wav2vec2 style of masking, allows overlap
                custom: no overlap, therefore bigger masking ratio 
          mask_prob:
            The probability of selecting choosing one frame as the start index
          mask_length:
            The length of each mask
          mask_selection:
            How to determine the length of the mask, see ``compute_mask_indices''
        """
        super().__init__()

        
        self.encoder = encoder
        self.encoder_dim = encoder_dim
            
        self.distillation_layer = distillation_layer
        # the frame ratio between the teacher and student
        # if larger than one, we are basically having more than one set of
        # codebooks for each frame
        self.num_codebooks= num_codebooks
        self.teacher_frame_ratio = teacher_frame_ratio 
        self.interpolate_teacher = interpolate_teacher
        self.distillation_delta = distillation_delta
        self.output_downsampling_factor = output_downsampling_factor
        
        if self.output_downsampling_factor > 1:
            self.downsample_output = SimpleDownsample(
                encoder_dim, downsample=self.output_downsampling_factor,
            )
        else:
            self.downsample_output = None
        
        self.codebook_loss_heads = nn.ModuleList()
        for cb, frame_ratio in zip(num_codebooks, teacher_frame_ratio):
            if cb > 0:
                codebook_loss_net = JointCodebookLoss(
                    predictor_channels=encoder_dim,
                    num_codebooks=cb * frame_ratio,
                    is_joint=False,
                    reduction="none",
                )
            else:
                codebook_loss_net = None
            self.codebook_loss_heads.append(codebook_loss_net)
        
        if len(self.codebook_loss_heads) == 0:
            self.codebook_loss_heads = None
        
        # whether to compute loss on all masked positions
        self.loss_only_mask = loss_only_mask

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, return_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        src_key_padding_mask = make_pad_mask(x_lens)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask) # (N,T,C)
        
        if self.output_downsampling_factor >= 2:
            encoder_out = encoder_out.transpose(0,1)  # (T,N,C)
            encoder_out = self.downsample_output(encoder_out)
            encoder_out = encoder_out.transpose(0,1)  # (N,T,C)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                encoder_out_lens = (encoder_out_lens + 1) // 2

        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
        
        masked_indices = None
        if return_mask:
            return encoder_out, encoder_out_lens, masked_indices
        else:
            return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        codebook_indexes: list[torch.Tensor] = None,
        at_targets: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, T), the input audio waveform.
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          codebook_indexes:
            Codebook indexes of teacher embeddings
          mask:
            If we perform w2v2 style of masking over the fbank frames
            
        Returns:
          Return the codebook loss
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert codebook_indexes is not None or at_targets is not None
        
        # Compute encoder outputs
        encoder_out, encoder_out_lens, mask_indices = self.forward_encoder(x, x_lens, return_mask=True)
            
        cb_losses = []
        if self.codebook_loss_heads is not None:
            for i, cb_loss_net in enumerate(self.codebook_loss_heads):
                cb_indexes = codebook_indexes[i]
                if cb_indexes is not None and cb_loss_net is not None:
                    codebook_loss = self.forward_codebook_loss(
                        encoder_out,
                        encoder_out_lens,
                        cb_indexes,
                        cb_loss_net=cb_loss_net,
                        teacher_frame_ratio=self.teacher_frame_ratio[i],
                        distillation_delta=self.distillation_delta[i],
                        reduction="none"
                    )
                    if self.loss_only_mask and mask_indices is not None:
                        # downsample the mask 
                        cur_mask_indices = nn.functional.avg_pool1d(mask_indices, 4) >= 0.5
                        assert cur_mask_indices.size(1) >= codebook_loss.size(1)
                        mask_indices = cur_mask_indices[:, :codebook_loss.size(1)].float()
                        codebook_loss = codebook_loss * cur_mask_indices
                    codebook_loss = codebook_loss.sum(dim=1) # (B,)    
                else:
                    codebook_loss = 0.0
                cb_losses.append(codebook_loss)
        
        return cb_losses

    def forward_codebook_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        codebook_indexes: torch.Tensor,
        cb_loss_net: torch.nn.Module,
        teacher_frame_ratio: int,
        distillation_delta: int,
        reduction: str = "sum",
    ):
        # align the encoder features with the codebook indexes
        
        # check if we need to upsample the targets to match the encoder output length
        if round(encoder_out.shape[1] / codebook_indexes.shape[1]) > teacher_frame_ratio:
            upsample_ratio = round(encoder_out.shape[1] / codebook_indexes.shape[1])
            codebook_indexes = codebook_indexes.repeat_interleave(upsample_ratio, dim=1)

        if self.interpolate_teacher:
            codebook_indexes = self.interpolate_codebook_indexes(
                encoder_out, codebook_indexes
            )
        else:
            if codebook_indexes.shape[1] != encoder_out.shape[1]:
                # align the codebook indexes to the frame rate of the student encoder out
                codebook_indexes = self.concat_successive_codebook_indexes(
                    encoder_out, codebook_indexes, ratio=teacher_frame_ratio
                )
                
        # the delta is associated with the frame-rate of the student encoder
        # so a bigger delta maybe necessary for 50Hz student encoder
        if distillation_delta > 0:
            codebook_indexes = codebook_indexes[:,:-distillation_delta, :]
            encoder_out = encoder_out[:, distillation_delta:, :]
            truncated_padding_mask = make_pad_mask(encoder_out_lens - distillation_delta)
            codebook_indexes = codebook_indexes.masked_fill(truncated_padding_mask.unsqueeze(-1), value=-100)
            
        # compute the loss
        N,T,_ = encoder_out.shape
        codebook_loss = cb_loss_net(encoder_out.float(), codebook_indexes)
        codebook_loss = codebook_loss.reshape(N,T,-1)
        num_cb = codebook_loss.size(-1) # this is the equivalent number of codebooks
        
        # normalize the loss by the number of codebooks
        if reduction == "sum":
            codebook_loss = codebook_loss.sum(dim=(1,2)) / num_cb # (B,)
        elif reduction == "none":
            codebook_loss = codebook_loss.sum(dim=2) / num_cb # (B,T)
        else:
            raise NotImplementedError()
        
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
                ],
                dim=1
            )
        assert codebook_indexes.size(1) == middle_layer_output.size(1) * ratio
        
        # Handling issue 2.
        codebook_indexes = codebook_indexes.reshape(N, t_expected, C * ratio)
        assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
        return codebook_indexes
