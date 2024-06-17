# Copyright    2021-2023  Xiaomi Corp.        (authors: Xiaoyu Yang,
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
import math
import random
from typing import List, Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from lhotse.dataset import SpecAugment
from lhotse.dataset.signal_transforms import time_warp as time_warp_impl

from icefall.utils import AttributeDict, make_pad_mask


class AudioTaggingModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        encoder_dim: int = 384,
        num_events: int = 527,
    ):
        """An audio tagging model

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
          encoder_dim:
            Dimension of the encoder.
          num_event:
            The number of classes.
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        )

        self.segment_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        )

        # for multi-class classification, reduction="none" for frame level
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
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
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        target: torch.Tensor,
        segment_length: int = 10,
        use_spec_aug: bool = False,
        use_time_warp: bool = False,
        use_time_mask: bool = False,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
        num_frame_masks: int = 10,
        features_mask_size: int = 27,
        num_feature_masks: int = 2,
        frames_mask_size: int = 100,
        max_frames_mask_fraction: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          segment_length:
            The number of frames to be aggregated in each segment
        Returns:
          Return the binary crossentropy loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        N, T, C = x.shape

        if use_spec_aug and self.training:
            if use_time_warp:
                assert supervision_segments is not None
                # Apply time warping before duplicating, so both copies have the same time-warp
                x = time_warp(
                    x,
                    time_warp_factor=time_warp_factor,
                    supervision_segments=supervision_segments,
                )

            # Apply frequency masking on two copies respectively
            x1 = frequency_mask(
                x,
                features_mask_size=features_mask_size,
                num_feature_masks=num_feature_masks,
            )
            x2 = frequency_mask(
                x,
                features_mask_size=features_mask_size,
                num_feature_masks=num_feature_masks,
            )

            if use_time_mask:
                # Apply time masking on two copies respectively
                # The operation also returns the mask indices
                x1, mask_indexes1 = time_mask(
                    x1,
                    num_frame_masks=num_frame_masks,
                    frames_mask_size=frames_mask_size,
                    max_frames_mask_fraction=max_frames_mask_fraction,
                )
                x2, mask_indexes2 = time_mask(
                    x2,
                    num_frame_masks=num_frame_masks,
                    frames_mask_size=frames_mask_size,
                    max_frames_mask_fraction=max_frames_mask_fraction,
                )

            x = torch.cat([x1, x2], dim=0)
        else:
            x = x.repeat(2, 1, 1)
            mask_indexes1 = None
            mask_indexes2 = None

        x_lens = x_lens.repeat(2)
        # Compute encoder outputs from the duplicated batch
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        # Forward the audio tagging module, frame level
        clip_logits = self.forward_audio_tagging(
            encoder_out=encoder_out, encoder_out_lens=encoder_out_lens, frame_level=False,
        )  # (2*N, num_classes)

        # 1. compute the clip-level main AT loss with ground truth target
        loss = self.criterion(clip_logits, target.repeat(2, 1)).sum()  # (2 * N, 527)

        # 2. compute the clip-level co-training loss
        if self.training:
            clip_level_co_training_target = torch.cat(
                [clip_logits[N:, :].detach(), clip_logits[:N, :].detach()]
            ).sigmoid()  # exchange target
            clip_level_co_training_loss = self.criterion(clip_logits, clip_level_co_training_target) # (2*N,)
            clip_level_co_training_loss = clip_level_co_training_loss.sum()
        else:
            clip_level_co_training_loss = 0.0

        padding_mask = make_pad_mask(encoder_out_lens)

        # 3. compute the segment-level aux AT loss with ground truth target
        logits = self.segment_classifier(encoder_out)  # (2*N, T, num_classes)
        num_segments = logits.size(1) // segment_length
        segment_level_logits = logits[:, :num_segments * segment_length, :]
        segment_level_logits = segment_level_logits.reshape(2 * N, num_segments, segment_length, -1)
        segment_level_logits = segment_level_logits.sum(dim=2) # (2 * N, T//4, 527)

        segment_level_at_loss = self.criterion(segment_level_logits, target.unsqueeze(dim=1).repeat(2, num_segments, 1)) 
        segment_level_at_loss = segment_level_at_loss.sum() / num_segments
        
        # 4. compute the segment-level co-training loss
        if self.training:
            # exchange target
            segment_level_co_training_target = torch.cat(
                [segment_level_logits[N:].detach(), segment_level_logits[:N].detach()]
            ).sigmoid()  

            segment_level_co_training_loss = self.criterion(
                segment_level_logits, segment_level_co_training_target
            )

            # mask the co-training loss at padding positions
            segment_padding_mask = nn.functional.max_pool1d(padding_mask.float(), segment_length)
            segment_padding_mask = segment_padding_mask[:, :num_segments]
            segment_level_co_training_loss.masked_fill_(segment_padding_mask.bool().unsqueeze(dim=-1), 0.0)

            # normalize the co-training loss
            segment_level_co_training_loss = segment_level_co_training_loss.sum() / num_segments
        else:
            segment_level_co_training_loss = 0.0

        return loss / 2.0, segment_level_at_loss / 2.0, segment_level_co_training_loss / 2.0, clip_level_co_training_loss / 2.0

    def forward_audio_tagging(
        self, encoder_out, encoder_out_lens, frame_level: bool = True
    ):
        """
        Args:
          encoder_out:
            A 3-D tensor of shape (N, T, C).
          encoder_out_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          frame_level:
            If true, do not average over the entier clip and return the frame level
            logits

        Returns:
          If frame_level==False, a 2-D tensor of shape (N, num_classes).
          else a 3-D tensor of shape (N, T, num_classes)
        """
        logits = self.classifier(encoder_out)  # (N, T, num_classes)
        if frame_level:
            return logits
        padding_mask = make_pad_mask(encoder_out_lens)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)  # mask the padding frames
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(
            logits
        )  # normalize the logits

        return logits


def time_warp(
    features: torch.Tensor,
    p: float = 0.9,
    time_warp_factor: Optional[int] = 80,
    supervision_segments: Optional[torch.Tensor] = None,
):
    if time_warp_factor is None or time_warp_factor < 1:
        return features
    assert (
        len(features.shape) == 3
    ), "SpecAugment only supports batches of single-channel feature matrices."
    features = features.clone()
    if supervision_segments is None:
        # No supervisions - apply spec augment to full feature matrices.
        for sequence_idx in range(features.size(0)):
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            features[sequence_idx] = time_warp_impl(
                features[sequence_idx], factor=time_warp_factor
            )
    else:
        # Supervisions provided - we will apply time warping only on the supervised areas.
        for sequence_idx, start_frame, num_frames in supervision_segments:
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            end_frame = start_frame + num_frames
            features[sequence_idx, start_frame:end_frame] = time_warp_impl(
                features[sequence_idx, start_frame:end_frame], factor=time_warp_factor
            )

    return features


def frequency_mask(
    features: torch.Tensor,
    p: float = 0.9,
    features_mask_size: int = 27,
    num_feature_masks: int = 10,
):
    # Apply frequency masking to a batch of features
    assert (
        len(features.shape) == 3
    ), "SpecAugment only supports batches of single-channel feature matrices."
    features = features.clone()
    for sequence_idx in range(features.size(0)):
        if random.random() > p:
            # Randomly choose whether this transform is applied
            continue
        feat = features[sequence_idx]
        mean = feat.mean()
        # Frequency masking
        feat, _ = mask_along_axis_optimized(
            feat,
            mask_size=features_mask_size,
            mask_times=num_feature_masks,
            mask_value=mean,
            axis=2,
        )
        features[sequence_idx] = feat

    return features


def time_mask(
    features: torch.Tensor,
    p: float = 0.9,
    num_frame_masks: int = 10,
    frames_mask_size: int = 100,
    max_frames_mask_fraction: float = 0.15,
):
    # apply time mask to a batch of features
    assert (
        len(features.shape) == 3
    ), "SpecAugment only supports batches of single-channel feature matrices."
    features = features.clone()
    mask_indexes = torch.zeros(features.size(0), features.size(1)).to(features.device)
    for sequence_idx in range(features.size(0)):
        if random.random() > p:
            # Randomly choose whether this transform is applied
            continue
        feat = features[sequence_idx]
        mean = feat.mean()
        # Time masking
        max_tot_mask_frames = max_frames_mask_fraction * feat.size(0)
        num_frame_masks = min(
            num_frame_masks,
            math.ceil(max_tot_mask_frames / frames_mask_size),
        )
        max_mask_frames = min(frames_mask_size, max_tot_mask_frames // num_frame_masks)
        feat, mask_index = mask_along_axis_optimized(
            feat,
            mask_size=max_mask_frames,
            mask_times=num_frame_masks,
            mask_value=mean,
            axis=1,
        )
        features[sequence_idx] = feat
        mask_indexes[sequence_idx] = mask_index

    return features, mask_indexes

def mask_along_axis_optimized(
    features: torch.Tensor,
    mask_size: int,
    mask_times: int,
    mask_value: float,
    axis: int,
) -> torch.Tensor:
    """
    Apply Frequency and Time masking along axis.
    Frequency and Time masking as described in the SpecAugment paper.

    :param features: input tensor of shape ``(T, F)``
    :mask_size: the width size for masking.
    :mask_times: the number of masking regions.
    :mask_value: Value to assign to the masked regions.
    :axis: Axis to apply masking on (1 -> time, 2 -> frequency)
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported!")

    mask_indexes = torch.zeros(features.size(0), device=features.device)
    
    features = features.unsqueeze(0)
    features = features.reshape([-1] + list(features.size()[-2:]))

    values = torch.randint(int(0), int(mask_size), (1, mask_times)) # the length of each mask
    min_values = torch.rand(1, mask_times) * (features.size(axis) - values) # the start index of each mask
    mask_starts = (min_values.long()).squeeze()
    mask_ends = (min_values.long() + values.long()).squeeze()

    if axis == 1:
        if mask_times == 1:
            features[:, mask_starts:mask_ends] = mask_value
            mask_indexes[mask_starts:mask_ends] = 1
            return features.squeeze(0), mask_indexes
        for (mask_start, mask_end) in zip(mask_starts, mask_ends):
            features[:, mask_start:mask_end] = mask_value
            mask_indexes[mask_start:mask_end] = 1
    else:
        if mask_times == 1:
            features[:, :, mask_starts:mask_ends] = mask_value
            mask_indexes[mask_starts:mask_ends] = 1
            return features.squeeze(0), mask_indexes
        for (mask_start, mask_end) in zip(mask_starts, mask_ends):
            features[:, :, mask_start:mask_end] = mask_value
            mask_indexes[mask_start:mask_end] = 1

    features = features.squeeze(0)
    return features, mask_indexes