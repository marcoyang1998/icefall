# Copyright    2025  Xiaomi Corp.        (authors: Xiaoyu Yang)
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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_multi_kd_multi_teacher import SimpleDownsample

from icefall.utils import make_pad_mask

class AudioTaggingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 768,
        num_events: int = 527,
        linear_softmax: bool = True,
        output_downsampling_factor: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        
        self.audio_tagging_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        ) # 527 classes
        self.linear_softmax = linear_softmax
        
        self.output_downsampling_factor = output_downsampling_factor
        if self.output_downsampling_factor > 1:
            self.downsample_output = SimpleDownsample(
                encoder_dim, downsample=self.output_downsampling_factor,
            )
        else:
            self.downsample_output = None
        
    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, freeze_encoder: bool=False,
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
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        with torch.set_grad_enabled((not freeze_encoder) and self.training):
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

        return encoder_out, encoder_out_lens
    
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        at_targets: torch.Tensor,
        freeze_encoder: bool = False,
    ):
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens, freeze_encoder=freeze_encoder)
        
        if self.linear_softmax:
            at_loss = self.forward_audio_tagging_linear_softmax(encoder_out, encoder_out_lens, at_targets, return_logits=False)
        else:
            at_loss = self.forward_audio_tagging(encoder_out, encoder_out_lens, at_targets, return_logits=False)
            
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