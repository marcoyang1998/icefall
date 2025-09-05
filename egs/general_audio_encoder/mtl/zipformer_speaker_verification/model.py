# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
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

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
from speechbrain.nnet.losses import AdditiveAngularMargin, LogSoftmaxWrapper

from icefall.utils import make_pad_mask


class SpeakerVerificationModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        encoder_dim: int,
        num_channels: int = 512,
        speaker_embed_dim: int = 192,
        num_speakers: int = 5994,
        use_aam: bool = True,
        margin: float = 0.2,
        normalize_fbank: bool = False,
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
        
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.num_channels = num_channels
        
        self.sv_module = ECAPA_TDNN(
            input_size=encoder_dim,
            channels=[num_channels] * 4 + [num_channels * 3], # 例如: [1024, 1024, 1024, 1024, 3072]
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=speaker_embed_dim
        )
        self.use_aam = use_aam
        if self.use_aam:
            self.aam = AdditiveAngularMargin(
                margin=margin,
            )
            self.loss = LogSoftmaxWrapper(self.aam)
            self.classifier = Classifier(input_size=speaker_embed_dim, out_neurons=num_speakers)
        else:
            self.classifier = nn.Linear(speaker_embed_dim, num_speakers)
        self.normalize_fbank = normalize_fbank

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
        # normalise fbank (utterance level)
        if self.normalize_fbank:
            x = self._normalize_fbank(x, x_lens)
        
        with torch.set_grad_enabled((not freeze_encoder) and self.training):
            x, x_lens = self.encoder_embed(x, x_lens)
            src_key_padding_mask = make_pad_mask(x_lens)
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
            encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
            encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
        
        # if an extra downsample is placed after the encoder
        # if self.encoder_downsample is not None:
        #     encoder_out = encoder_out.permute(1, 0, 2)
        #     encoder_out = self.encoder_downsample(encoder_out)
        #     encoder_out = encoder_out.permute(1, 0, 2)
        #     encoder_out_lens = (encoder_out_lens + 1 ) // 2

        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        labels: torch.Tensor,        
        freeze_encoder: bool = False,
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

        device = x.device

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(
            x, 
            x_lens, 
            freeze_encoder=freeze_encoder
        )

        sv_loss = self.forward_speaker_verification(
            encoder_out, encoder_out_lens, labels=labels
        )
        
        return sv_loss
    
    def compute_sv_loss(self, embed, labels):
        """Compute SV loss

        Args:
            embed (_type_): speaker embedding
            labels (_type_): speaker identities

        Returns:
            _type_: _description_
        """
        if self.use_aam:
            logits = self.classifier(embed)
            loss = self.loss(logits, labels)
        else:
            logits = self.classifier(embed) # (N, num_speakers)
            logits = logits.squeeze(1)
            labels = F.one_hot(labels, num_classes=logits.size(1)).to(logits.dtype).squeeze(1)
            loss = F.cross_entropy(logits, labels, reduction="sum") / logits.size(0)
        return loss
    
    def compute_speaker_embedding(self, encoder_out, encoder_out_lens):
        """Compute speaker embeddings.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        Returns:
          speaker_embeddings:
            Speaker embeddings of shape (N, speaker_embed_dim).
        """
        encoder_out_lens = encoder_out_lens / encoder_out_lens.max()
        speaker_embed = self.sv_module(encoder_out, encoder_out_lens)
        return speaker_embed

    def forward_speaker_verification(self, encoder_out, encoder_out_lens, labels):
        """Compute speaker verification loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        embed = self.compute_speaker_embedding(encoder_out, encoder_out_lens)
        loss = self.compute_sv_loss(embed, labels)
        return loss
    
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