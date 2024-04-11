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
from lhotse.dataset.signal_transforms import mask_along_axis_optimized
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

        # for multi-class classification
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

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
          target:
            The ground truth label of audio events, could be many hot (N, num_classes)
        Returns:
          Return the binary crossentropy loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        N, T, C = x.shape

        # import pdb; pdb.set_trace()
        if use_spec_aug and self.training:
            if use_time_warp:
                assert supervision_segments is not None
                # Apply time warping before duplicating
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
                x1 = time_mask(
                    x1,
                    num_frame_masks=num_frame_masks,
                    frames_mask_size=frames_mask_size,
                    max_frames_mask_fraction=max_frames_mask_fraction,
                )
                x2 = time_mask(
                    x2,
                    num_frame_masks=num_frame_masks,
                    frames_mask_size=frames_mask_size,
                    max_frames_mask_fraction=max_frames_mask_fraction,
                )

            x = torch.cat([x1, x2], dim=0)
        else:
            x = x.repeat(2, 1, 1)

        x_lens = x_lens.repeat(2)
        # Compute encoder outputs from the duplicated batch
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        # Forward the audio tagging module
        logits = self.forward_audio_tagging(
            encoder_out=encoder_out, encoder_out_lens=encoder_out_lens
        )  # (2*N, num_classes)

        # compute the co-training loss
        co_training_target = torch.cat(
            [logits[N:, :].detach(), logits[:N, :].detach()]
        ).sigmoid()  # exchange target
        co_training_loss = self.criterion(logits, co_training_target)

        loss = self.criterion(logits, target.repeat(2, 1))

        return loss / 2.0, co_training_loss / 2.0

    def forward_audio_tagging(self, encoder_out, encoder_out_lens):
        """
        Args:
          encoder_out:
            A 3-D tensor of shape (N, T, C).
          encoder_out_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          A 3-D tensor of shape (N, num_classes).
        """
        logits = self.classifier(encoder_out)  # (N, T, num_classes)
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
        feat = mask_along_axis_optimized(
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
        feat = mask_along_axis_optimized(
            feat,
            mask_size=max_mask_frames,
            mask_times=num_frame_masks,
            mask_value=mean,
            axis=1,
        )
        features[sequence_idx] = feat

    return features
