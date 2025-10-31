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
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_quantization.prediction import JointCodebookLoss

from model_multi_kd_w2v2_mask import index_put, compute_mask_indices, compute_mask_indices_block

from icefall.utils import make_pad_mask


class MAELoss(torch.nn.Module):
    def __init__(self, normalize_mode: str):
        super().__init__()
        # If True, normalise the target by frame
        assert normalize_mode in ["frame", "sample", "batch"]
        self.normalize_mode = normalize_mode

    def forward(self, pred: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
        if self.normalize_mode == "frame": # adopted by Dasheng
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        elif self.normalize_mode == "sample":
            mean = target.mean(dim=(1, 2), keepdim=True) # per sample
            var = target.var(dim=(1, 2), keepdim=True)
            target = (target - mean) / (var + 1e-6)**.5
        elif self.norm_by_frame == 'batch':
            mean = target.mean()
            var = target.var()
            target = (target - mean) / (var + 1.e-6)**.5
        
        # compute the MSE loss
        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch, normalized by the fbank dim
        return loss

class MultiKDModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        encoder_dim: int,
        decoder: nn.Module,
        decoder_dim: int,
        decoder_input_dim: int,
        num_codebooks: int=8,
        distillation_layer: int=9,
        distillation_delta: int=0,
        teacher_frame_ratio: int = 2,
        interpolate_teacher: bool = False,
        n_mels: int = 128,
        num_events: int = 527,
        mae_loss_norm: str = "sample",
        mae_downsample_factor: int = 4,
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
        loss_only_mask: bool = False,
    ):
        """A model that performs MVQ KD pre-training with additional MAE loss.

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
            The number of codebooks used in the target
          distillation_layer:
            Use which layer to do MVQ pre-training
          distillation_delta:
            How many frames to delay the alignment between the model and the target frames.
            Should be zero for non-streaming models, and a positive number for streaming models
          teacher_frame_ratio:
            The frame rate ratio between the target and the model output
          mae_loss_norm:
            The normalization mode of the MAE loss
          mae_downsample_factor:
            The ratio between of final output frame rate after the decoder and the encoder input
          mask_prob:
            The probability of selecting choosing one frame as the start index
          mask_length:
            The length of each mask
          mask_selection:
            How to determine the length of the mask, see ``compute_mask_indices''
        """
        super().__init__()

        
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.decoder = decoder
        self.decoder_dim = decoder_dim
        
        self.fbank_dim = n_mels
        self.mae_downsample_factor = mae_downsample_factor
        self.decoder_embed = nn.Linear(encoder_dim, decoder_input_dim) # projecting encoder_out to decoder dim
        self.decoder_pred = nn.Linear(decoder_dim, n_mels * mae_downsample_factor) # we are predicting 4 fbank frames per decoder frame
            
        # mvq distillation
        self.distillation_layer = distillation_layer
        # the frame ratio between the teacher and student
        # if larger than one, we are basically having more than one set of
        # codebooks for each frame
        self.num_codebooks= num_codebooks
        self.teacher_frame_ratio = teacher_frame_ratio 
        self.interpolate_teacher = interpolate_teacher
        self.distillation_delta = distillation_delta
        
        if num_codebooks > 0:
            self.codebook_loss_net = JointCodebookLoss(
                predictor_channels=encoder_dim,
                num_codebooks=num_codebooks * self.teacher_frame_ratio,
                is_joint=False,
                reduction="none",
            )
        else:
            self.codebook_loss_net = None
        
        self.mae_loss_norm = mae_loss_norm
        self.mae_loss = MAELoss(mae_loss_norm)
        
        self.audio_tagging_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        ) # 527 classes
        
        # masking related
        assert mask_mode in ["w2v2", "block"], f"Unseen mask mode: {mask_mode}"
        self.mask_mode = mask_mode
        
        self.mask_emb = nn.Parameter(torch.FloatTensor(n_mels).normal_()) 
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.min_masks = min_masks
        
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        
        self.loss_only_mask = loss_only_mask

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
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
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_decoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ):
        """Compute the output of decoder

        Args:
            x (torch.Tensor): (N,T,C)
            x_lens (torch.Tensor): (N,)
        """
        x = self.decoder_embed(x) # N,T,C
        
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1,0,2) # T,N,C
        
        decoder_out, decoder_out_lens = self.decoder(x, x_lens, src_key_padding_mask)
        decoder_out = decoder_out.permute(1,0,2) # N,T,C
        
        return decoder_out, decoder_out_lens
        
    def forward_mae_loss(
        self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor, target: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        # compute the MAE loss
        decoder_out, decoder_out_lens = self.forward_decoder(
            encoder_out, encoder_out_lens,
        )
        pred = self.decoder_pred(decoder_out) # map to 4 * fbank dim
        N,T,_ = pred.shape
        pred = pred.reshape(N, -1, self.fbank_dim)
        
        assert pred.shape[2] == target.shape[2]
        target = self.truncate_target(pred, target)
        loss = self.mae_loss(pred, target) # (N,T)
        
        # mask the loss on padding positions to zero
        padding_mask = ~ make_pad_mask(decoder_out_lens * self.mae_downsample_factor)
        loss = loss * padding_mask
        if reduction == "sum":
            loss = loss.sum(dim=1) # (N,)
        return loss
        
    @staticmethod
    def truncate_target(pred: torch.Tensor, target: torch.Tensor):
        # truncate the target on both sides for better alignment
        # Only consider the cases where the target is longer
        # pred: (N,T,C)
        # target: (N,T,C)
        assert target.shape[1] >= pred.shape[1]
        if target.shape[1] == pred.shape[1]:
            return target
        diff = target.shape[1] - pred.shape[1]
        if diff == 1:
            target = target[:, :-1, :] # throw the last frame
        else:
            left = diff // 2
            right = diff - left
            target = target[:, left:-right, :] # trim on both sides
        assert target.shape[1] == pred.shape[1]
        
        return target
        
    
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        fbank_target: torch.Tensor,
        codebook_indexes: torch.Tensor = None,
        at_targets: torch.Tensor = None,
        mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          fbank_target:
            The original fbank features
          codebook_indexes:
            Codebook indexes of teacher embeddings
          mask:
            If we perform w2v2 style of masking over the fbank frames
            
        Returns:
          Return the codebook loss and the mae loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert codebook_indexes is not None or at_targets is not None

        # apply masking
        if self.training and mask:
            padding_mask = make_pad_mask(x_lens)
            
            # apply masking to the fbank features
            x, mask_indices = self.apply_mask(
                x.clone(),
                padding_mask=padding_mask
            ) # (N,T,C), (N,T)
        else:
            mask_indices = None
        
        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
            
        # Compute codebook loss
        if codebook_indexes is not None and self.codebook_loss_net is not None:
            codebook_loss = self.forward_codebook_loss(
                encoder_out, encoder_out_lens, codebook_indexes, reduction="none"
            )
            if self.loss_only_mask and mask_indices is not None:
                # downsample the mask 
                mask_indices = nn.functional.avg_pool1d(mask_indices, 4) >= 0.5
                assert mask_indices.size(1) >= codebook_loss.size(1)
                mask_indices = mask_indices[:, :codebook_loss.size(1)].float()
                codebook_loss = codebook_loss * mask_indices
            codebook_loss = codebook_loss.sum(dim=1) # (B,)    
        else:
            codebook_loss = None
            
        # Compute audio tagging loss (if needed)
        if at_targets is not None:
            at_loss = self.forward_audio_tagging(encoder_out, encoder_out_lens, at_targets, return_logits=False)
        else:
            at_loss = None
        
        mae_loss = self.forward_mae_loss(encoder_out, encoder_out_lens, fbank_target, reduction="none") # (N,T)
        # we only compute the fbank reconstruction loss on masked positions, regardless of loss_only_mask
        if mask_indices is not None:
            diff = mask_indices.shape[1] - mae_loss.shape[1]
            if diff > 0:
                mask_indices = mask_indices[:, diff//2: diff//2 + mae_loss.shape[1]]
            mae_loss = mae_loss * mask_indices
        mae_loss = mae_loss.sum(dim=-1) # (N)
        
        return codebook_loss, at_loss, mae_loss

    def forward_codebook_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        codebook_indexes: torch.Tensor,
        reduction: str = "sum"
    ):
        # align the encoder features with the codebook indexes
        if self.interpolate_teacher:
            codebook_indexes = self.interpolate_codebook_indexes(
                encoder_out, codebook_indexes
            )
        else:
            if codebook_indexes.shape[1] != encoder_out.shape[1]:
                # align the codebook indexes to the frame rate of the student encoder out
                codebook_indexes = self.concat_successive_codebook_indexes(
                    encoder_out, codebook_indexes, ratio=self.teacher_frame_ratio
                )
                
        # the delta is associated with the frame-rate of the encoder
        # so a bigger delta maybe necessary for 50Hz student encoder
        if self.distillation_delta > 0:
            codebook_indexes = codebook_indexes[:,:-self.distillation_delta, :]
            encoder_out = encoder_out[:, self.distillation_delta:, :]
            truncated_padding_mask = make_pad_mask(encoder_out_lens - self.distillation_delta)
            codebook_indexes = codebook_indexes.masked_fill(truncated_padding_mask.unsqueeze(-1), value=-100)
            
        N,T,_ = encoder_out.shape
        codebook_loss = self.codebook_loss_net(encoder_out.float(), codebook_indexes)
        codebook_loss = codebook_loss.reshape(N,T,-1)
        num_cb = codebook_loss.size(-1)
        # normalize the loss by the number of codebooks
        if reduction == "sum":
            codebook_loss = codebook_loss.sum(dim=(1,2)) / num_cb # (B,)
        elif reduction == "none":
            codebook_loss = codebook_loss.sum(dim=2) / num_cb # (B,T)
        else:
            raise NotImplementedError()
        
        return codebook_loss

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
