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

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
from speechbrain.nnet.losses import AdditiveAngularMargin, LogSoftmaxWrapper

from icefall.utils import make_pad_mask


class AMSoftmaxLoss(nn.Module):

    def __init__(self, hidden_dim, speaker_num, s=15.0, m=0.4, **kwargs):
        '''
        AM Softmax Loss
        '''
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.speaker_num = speaker_num
        self.W = torch.nn.Parameter(torch.randn(hidden_dim, speaker_num), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x_BxH, labels_B):
        '''
        x shape: (B, H)
        labels shape: (B)
        '''
        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        
        W = F.normalize(self.W, dim=0)

        x_BxH = F.normalize(x_BxH, dim=1)

        wf = torch.mm(x_BxH, W)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels_B]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels_B)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

# this is borrowed from s3prl
class AAMSoftmaxLoss(nn.Module):
    def __init__(self, hidden_dim, speaker_num, s=15, m=0.3, easy_margin=False, **kwargs):
        super(AAMSoftmaxLoss, self).__init__()
        import math

        self.test_normalize = True
        
        self.m = m
        self.s = s
        self.speaker_num = speaker_num
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(speaker_num, hidden_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x_BxH, labels_B):

        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        
        # cos(theta)
        cosine = F.linear(F.normalize(x_BxH), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels_B.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, labels_B)
        return loss

class SpeakerVerificationModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        encoder_dim: int,
        num_encoder_layers: int = 11,
        use_weighted_sum: bool = False,
        num_channels: int = 512,
        speaker_embed_dim: int = 192,
        num_speakers: int = 5994,
        use_aam: bool = True,
        margin: float = 0.2,
        scale: float = 30.0,
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
            self.classifier = AAMSoftmaxLoss(
                hidden_dim=speaker_embed_dim,
                speaker_num=num_speakers, 
                m=margin,
                s=scale,
            )
        else:
            self.classifier = nn.Linear(speaker_embed_dim, num_speakers)
        
        # layer-wise weights for middle layers
        self.use_weighted_sum = use_weighted_sum
        if use_weighted_sum:
            self.weights = torch.nn.Parameter(torch.ones(num_encoder_layers) / num_encoder_layers, requires_grad=True)
        else:
            self.weights = None
        
        self.normalize_fbank = normalize_fbank

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, freeze_encoder: bool=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs. Either return the last layer of the weighted-sum of intermediate layers
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
            encoder_out, encoder_out_lens, middle_out = self.encoder(x, x_lens, src_key_padding_mask, return_middle_out=True)
            encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        
        # compute weighted sum for intermediate layers
        if self.use_weighted_sum:
            middle_out = [out.permute(1, 0, 2) for out in middle_out] # each (N, T, C)
            middle_out = torch.stack(middle_out, dim=0) # (num_layers, N, T, C)
            weights = F.softmax(self.weights, dim=0) # (num_layers,)
            weights = weights.view(-1, 1, 1, 1) # (num_layers, 1, 1, 1)
            encoder_out = (middle_out * weights).sum(dim=0) # (N, T, C)
        
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

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
            freeze_encoder=freeze_encoder,
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
            embed = embed.squeeze(1) # (N, speaker_embed_dim)
            labels = labels.squeeze(1) # (N)
            loss = self.classifier(embed, labels)
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
        with torch.cuda.amp.autocast(enabled=False):
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