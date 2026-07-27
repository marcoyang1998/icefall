# Copyright    2026  University of Cambridge    (authors: Xiaoyu Yang)
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
from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F

from icefall.utils import add_sos, make_pad_mask

class MultiTaskModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_downsample: Optional[nn.Module] = None,
        attention_decoder: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        use_attention_decoder: bool = False,
        num_events: int = 527,
        normalize_fbank: bool = False,
        weighted_sum: bool = True,
        weighted_sum_normalize: bool = True,
        temperature: float = 1.0,
        topk_layers: int = -1,
        linear_softmax: bool = False,
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
          normalize_fbank:
            If true, the input fbank features is normalized to zero mean and unit variance utterance-wise
        """
        super().__init__()

        # assert (
        #     use_transducer or use_ctc
        # ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_downsample = encoder_downsample

        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = nn.Linear(encoder_dim, vocab_size)
            
            self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )
        
        self.use_attention_decoder = use_attention_decoder
        self.attention_decoder = attention_decoder
        
        self.audio_tagging_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        ) # 527 classes
        
        self.normalize_fbank = normalize_fbank
        self.linear_softmax = linear_softmax
        
        # weighted sum
        self.weighted_sum = weighted_sum
        self.topk_layers = topk_layers
        self.weighted_sum_normalize = weighted_sum_normalize
        self.temperature = temperature
        if self.weighted_sum:
            if topk_layers > 0:
                zero_init = torch.cat([torch.zeros(topk_layers)])
                self.layer_weights = torch.nn.Parameter(zero_init, requires_grad=True)
            else:
                self.num_encoder_layers = sum(self.encoder.num_encoder_layers)
                zero_init = torch.cat([torch.zeros(self.num_encoder_layers)])
                self.layer_weights = torch.nn.Parameter(zero_init, requires_grad=True)
        else:
            self.layer_weights = None

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, freeze_encoder: bool=False, return_weighted_out: bool = True,
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
        
        # normalise fbank (utterance level)
        if self.normalize_fbank:
            x = self._normalize_fbank(x, x_lens)
        
        with torch.set_grad_enabled((not freeze_encoder) and self.training):
            x, x_lens = self.encoder_embed(x, x_lens)
            src_key_padding_mask = make_pad_mask(x_lens)
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
            encoder_out, encoder_out_lens, middle_out = self.encoder(x, x_lens, src_key_padding_mask, return_middle_out=True)
            encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
        
        if return_weighted_out and self.layer_weights is not None:
            middle_out = [m.permute(1,0,2) for m in middle_out] # [(N,T,C), ...]
            middle_out = torch.stack(middle_out, dim=0) # (num_layers, N, T, C)
            if self.flow_topk_layers > 0:
                middle_out = middle_out[-self.flow_topk_layers:, :, :, :] # (topk, N, T, C)
            if random.random() < 0.01:
                with torch.no_grad():
                    vector_norms = middle_out.norm(p=2, dim=-1)
                    layer_magnitudes = vector_norms.mean(dim=(1, 2))
                logging.info(f"Layer wise magnitudes before norm: {np.round(layer_magnitudes.data.cpu().numpy(), 2)}")
                
            if self.flow_weighted_sum_normalize:
                middle_out = F.layer_norm(middle_out, (middle_out.shape[-1],)) # normalize on the feature dimension
            
            layer_weights = F.softmax(self.layer_weights / self.flow_temperature, dim=0).view(-1,1,1,1) # (num_layers,)
            if random.random() < 0.01:
                logging.info(f"Layer weights: {np.round(layer_weights.squeeze().data.cpu().numpy(), 4)}")
            weighted_sum_out = (middle_out * layer_weights).sum(dim=0) # (N,T,C) 
            if weighted_sum_out.shape[1] % 2 == 1:
                weighted_sum_out = F.pad(weighted_sum_out, (0, 0, 0, 1), mode="replicate")
            weighted_sum_out = F.avg_pool1d(weighted_sum_out.permute(0,2,1), 2, padding=1).permute(0,2,1) # (N,T/2,C)
            weighted_sum_out = weighted_sum_out[:, :encoder_out.size(1), :] # align with encoder_out
        else:
            weighted_sum_out = None
        
        # if an extra downsample is placed after the encoder
        if self.encoder_downsample is not None:
            encoder_out = encoder_out.permute(1, 0, 2)
            encoder_out = self.encoder_downsample(encoder_out)
            encoder_out = encoder_out.permute(1, 0, 2)
            encoder_out_lens = (encoder_out_lens + 1 ) // 2

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=encoder_out_lens,
            target_lengths=target_lengths,
            reduction="none",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="none",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="none",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        at_targets: torch.Tensor = None,
        freeze_encoder: bool = False,
        skip_asr: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        # assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)
        device = x.device

        # Compute encoder outputs
        encoder_out, encoder_out_lens, weighted_sum_out = self.forward_encoder(
            x, 
            x_lens, 
            freeze_encoder=freeze_encoder
        )

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer and not skip_asr:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(x.device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc and not skip_asr:
            # Compute CTC loss
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)
            
        if self.use_attention_decoder:
            attention_decoder_loss = self.attention_decoder.calc_att_loss(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                ys=y.to(device),
                ys_lens=y_lens.to(device),
            )
        else:
            attention_decoder_loss = torch.empty(0)
            
        if at_targets is not None:
            if self.linear_softmax:
                at_loss = self.forward_audio_tagging_linear_softmax(encoder_out, encoder_out_lens, at_targets, return_logits=False)
            else:
                at_loss = self.forward_audio_tagging(encoder_out, encoder_out_lens, at_targets, return_logits=False)
        else:
            at_loss = None
        
        return simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, at_loss
    
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
    
    @staticmethod
    def _normalize_fbank(x: torch.Tensor, x_lens: torch.Tensor, eps: float=1e-9):
        """
        x: (B, T, D) fbank 特征，已 padding 到同一 T
        x_lens: (B,) 每条样本的有效帧数 (int)
        """
        device = x.device
        B, T, D = x.shape

        # mask: (B, T, 1)
        mask = torch.arange(T, device=device).unsqueeze(0) < x_lens.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # (B, T, 1), bool

        lengths = x_lens.view(B, 1, 1).to(x.dtype)  # (B, 1, 1)

        # 均值
        sum_feats = (x * mask).sum(dim=1, keepdim=True)  # (B, 1, D)
        mean = sum_feats / lengths

        # 方差
        sum_sq = ((x - mean) * mask).pow(2).sum(dim=1, keepdim=True)
        std = torch.sqrt(sum_sq / lengths + eps)

        # 归一化
        x_norm = (x - mean) / (std + eps)
        # set masking positions to value 0
        x_norm = x_norm * mask

        return x_norm

