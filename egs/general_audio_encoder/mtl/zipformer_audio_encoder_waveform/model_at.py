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
from typing import Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from wav2vec2_utils import compute_mask_indices, compute_mask_indices_block, index_put, GradMultiply
from icefall.utils import make_pad_mask


class AudioTaggingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_embed: nn.Module,
        encoder_dim: int,
        encoder_downsample: Optional[nn.Module] = None,
        output_downsampling_factor: int = 2,
        conv_dim: int = 512,
        feature_grad_mult: float = 1.0,
        mask_mode: str = "w2v2",
        mask_prob: float = 0.65,
        mask_length: int = 10,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 2,
        mask_channel_prob: float = 0.0,
        mask_channel_length: int = 10,
        mask_channel_selection: str = "static",
        mask_channel_other: float = 0.0,
        num_events: int = 527,
        linear_softmax: bool = False,
    ):
        """A MVQ pretrained encoder

        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_downsample = encoder_downsample
        self.encoder_dim = encoder_dim
        self.output_downsampling_factor = output_downsampling_factor
        self.feature_grad_mult = feature_grad_mult
                        
        # masking related
        assert mask_mode in ["w2v2", "block"], f"Unseen mask mode: {mask_mode}"
        self.mask_mode = mask_mode
        
        self.mask_emb = nn.Parameter(torch.FloatTensor(conv_dim).normal_()) 
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.min_masks = min_masks
        
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        
        self.linear_softmax = linear_softmax
        
        self.audio_tagging_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        ) # 527 classes

    def forward_encoder_embed(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the features of the convolution frontend

        Args:
            x (torch.Tensor): input audio waveform
            x_lens (torch.Tensor): length of audio waveform

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output features and their lengths
        """
        if self.feature_grad_mult > 0:
            x, x_lens = self.encoder_embed(x, x_lens)
            if self.feature_grad_mult != 1.0:
                x = GradMultiply.apply(x, self.feature_grad_mult)
        else:
            with torch.no_grad():
                x, x_lens = self.encoder_embed(x, x_lens)
        return x, x_lens
    
    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, apply_mask: bool = True, freeze_encoder: bool=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 2-D tensor of shape (N, T). Audio waveform
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        
        with torch.set_grad_enabled((not freeze_encoder) and self.training):
            x, x_lens = self.forward_encoder_embed(x, x_lens)  # (N, T, C)
            assert x_lens is not None
            
            if self.training and apply_mask:
                padding_mask = make_pad_mask(x_lens)
                
                # apply masking to the fbank features
                x, mask_indices = self.apply_mask(
                    x.clone(),
                    padding_mask=padding_mask
                ) # (N,T,C), (N,T)
            else:
                mask_indices = None
            
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
            
            src_key_padding_mask = make_pad_mask(x_lens)
            encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask) # (N,T,C)
            
            encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
        
        # if an extra downsample is placed after the encoder
        if self.encoder_downsample is not None:
            encoder_out = encoder_out.permute(1, 0, 2)
            encoder_out = self.encoder_downsample(encoder_out)
            encoder_out = encoder_out.permute(1, 0, 2)
            encoder_out_lens = (encoder_out_lens + 1 ) // 2

        return encoder_out, encoder_out_lens, mask_indices

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        at_targets: torch.Tensor = None,
        freeze_encoder: bool = False,
        apply_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, T), the input audio waveform.
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
            
        Returns:
          audio tagging loss
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        # Compute encoder outputs
        encoder_out, encoder_out_lens, mask_indices = self.forward_encoder(
            x, 
            x_lens,
            apply_mask=apply_mask,
            freeze_encoder=freeze_encoder,
        )
        
        if self.linear_softmax:
            at_loss = self.forward_audio_tagging_linear_softmax(encoder_out, encoder_out_lens, at_targets, return_logits=False)
        else:
            at_loss = self.forward_audio_tagging(encoder_out, encoder_out_lens, at_targets, return_logits=False)
        
        return at_loss
    
    def forward_audio_tagging_linear_softmax(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        target: torch.Tensor = None,
        return_logits: bool = False,
    ):
        # target: (N, num_events)
        frame_logits = self.audio_tagging_proj(encoder_out)

        # --- Linear Softmax Pooling (Corrected Version) 开始 ---

        # 2. 将Logits转换为帧级别的概率 (激活值)
        # (N, T, num_classes)
        frame_probabilities = torch.sigmoid(frame_logits)
        
        # 3. 处理padding，将填充部分的概率设为0
        padding_mask = make_pad_mask(encoder_out_lens) # (N, T)
        expanded_padding_mask = padding_mask.unsqueeze(-1).expand_as(frame_probabilities)

        frame_probabilities = frame_probabilities.masked_fill(expanded_padding_mask, 0.0)

        # 4. 计算线性归一化权重 (不使用exp)
        # 沿时间维度求和，用于归一化
        # 添加一个小的epsilon防止除以零
        sum_over_time = torch.sum(frame_probabilities, dim=1, keepdim=True) + 1e-7
        
        # 权重就是归一化后的概率
        # (N, T, num_classes)
        linear_weights = frame_probabilities / sum_over_time

        # 5. 使用线性权重对原始的帧级别概率进行加权求和
        # (N, T, num_classes) * (N, T, num_classes) -> (N, T, num_classes)
        # 然后在时间维度上求和 -> (N, num_classes)
        clip_probabilities = torch.sum(linear_weights * frame_probabilities, dim=1)
        
        # --- Linear Softmax Pooling 结束 ---

        if return_logits: # 实际上返回的是概率
            return clip_probabilities
        
        # Compute loss, F.binary_cross_entropy does not support amp
        with torch.cuda.amp.autocast(enabled=False):
            at_loss = F.binary_cross_entropy(clip_probabilities.float(), target.float(), reduction="none")

        return at_loss
    
    def forward_audio_tagging(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        target: torch.Tensor = None,
        return_logits: bool = False,
    ):
        # target: (N, num_events)
        logits = self.audio_tagging_proj(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens) # (N,T)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits) # (N, num_events)
        if return_logits:
            return logits
        
        at_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        return at_loss
    
    
    def apply_mask(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mask according to the mask_mode, return the masked features and the masked positions

        Args:
            x (torch.Tensor): The input fbank features
            padding_mask (torch.Tensor, optional): The padding mask

        Returns:
            The masked fbank feature and the masked_indices, with masked positions as 1
        """
        # apply mask to the fbank features, two modes applicable
        if self.mask_mode == "w2v2":
            x, masked_indices = self.apply_mask_w2v2(x, padding_mask)
        elif self.mask_mode == "block":
            x, masked_indices = self.apply_mask_block(x, padding_mask)
        else:
            raise NotImplementedError()
        
        if random.random() > 0.97:
            logging.info(f"Apply {self.mask_mode} masking. A proportion of {masked_indices.sum()/masked_indices.numel():.2f} frames are masked")
        return x, masked_indices
    
    def apply_mask_block(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ):
        B,T,C = x.shape
        assert self.mask_prob > 0.0

        mask_indices = compute_mask_indices_block(
            shape=(B,T),
            padding_mask=padding_mask,
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
            min_masks=self.min_masks,
        ).to(x.device)
        
        x = index_put(x, mask_indices.bool(), self.mask_emb)

        return x, mask_indices
    
    def apply_mask_w2v2(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ):
        # this function is modified from fairseq: https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/models/wav2vec/wav2vec2.py#L429
        # The masked indices have value 1
        B, T, C = x.shape
        
        # we mask channel first, then mask timestamps
        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=False,
                min_space=1,
                require_same_masks=False,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            if random.random() > 0.98:
                logging.info(f"A proportion of {mask_channel_indices.sum()/mask_channel_indices.numel():.2f} feature dims are masked")
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                mask_type=self.mask_selection,
                mask_other=self.mask_other,
                min_masks=2, # fixed
                no_overlap=False,  # False
                min_space=1,  # 1
                require_same_masks=False,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
            mask_indices = mask_indices.float()
        else:
            mask_indices = None

        return x, mask_indices
    
if __name__=="__main__":
    B = 10
    C = 256
    mask_channel_indices = compute_mask_indices(
        (B, C),
        None,
        0.25,
        20,
        "static",
        0.0,
        no_overlap=False,
        min_space=1,
        require_same_masks=False,
    )
    mask_channel_indices = torch.from_numpy(mask_channel_indices)
    print(mask_channel_indices.sum()/mask_channel_indices.numel())
    
    pass