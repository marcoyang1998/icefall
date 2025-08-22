import bisect
import math
import random
from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch

from lhotse import CutSet
from lhotse.utils import Pathlike

__all__ = ["DiscretizedInputAugment"]


class DiscretizedInputAugment(torch.nn.Module):
    """
    SpecAugment performs three augmentations:
    - time warping of the token matrix
    - masking of ranges of tokens (frequency bands)
    - masking of ranges of frames (time)

    The current implementation works with batches, but processes each example separately
    in a loop rather than simultaneously to achieve different augmentation parameters for
    each example.
    """

    def __init__(
        self,
        token_type: str,
        time_warp_factor: Optional[int] = 80,
        num_token_masks: int = 2,
        tokens_mask_size: int = 27,
        num_frame_masks: int = 10,
        frames_mask_size: int = 100,
        max_frames_mask_fraction: float = 0.15,
        p=0.9,
    ):
        """
        SpecAugment's constructor.

        :param time_warp_factor: parameter for the time warping; larger values mean more warping.
            Set to ``None``, or less than ``1``, to disable.
        :param num_token_masks: how many token masks should be applied. Set to ``0`` to disable.
        :param tokens_mask_size: the width of the token mask (expressed in the number of masked token bins).
            This is the ``F`` parameter from the SpecAugment paper.
        :param num_frame_masks: the number of masking regions for utterances. Set to ``0`` to disable.
        :param frames_mask_size: the width of the frame (temporal) masks (expressed in the number of masked frames).
            This is the ``T`` parameter from the SpecAugment paper.
        :param max_frames_mask_fraction: limits the size of the frame (temporal) mask to this value times the length
            of the utterance (or supervision segment).
            This is the parameter denoted by ``p`` in the SpecAugment paper.
        :param p: the probability of applying this transform.
            It is different from ``p`` in the SpecAugment paper!
        """
        super().__init__()
        assert 0 <= p <= 1
        assert num_token_masks >= 0
        assert num_frame_masks >= 0
        assert tokens_mask_size > 0
        assert frames_mask_size > 0
        self.token_type = token_type
        self.time_warp_factor = time_warp_factor
        self.num_token_masks = num_token_masks
        self.tokens_mask_size = tokens_mask_size
        self.num_frame_masks = num_frame_masks
        self.frames_mask_size = frames_mask_size
        self.max_frames_mask_fraction = max_frames_mask_fraction
        self.p = p

    def forward(
        self,
        tokens: torch.Tensor,
        num_tokens: int = 2000,
        frequency_size: int = 80,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes SpecAugment for a batch of token matrices.

        Since the batch will usually already be padded, the user can optionally
        provide a ``supervision_segments`` tensor that will be used to apply SpecAugment
        only to selected areas of the input. The format of this input is described below.

        :param tokens: a batch of token matrices with shape ``(B, T)``.
        :param num_tokens: the number of discrete tokens.
        :param supervision_segments: an int tensor of shape ``(S, 3)``. ``S`` is the number of
            supervision segments that exist in ``tokens`` -- there may be either
            less or more than the batch size.
            The second dimension encoder three kinds of information:
            the sequence index of the corresponding token matrix in `tokens`,
            the start frame index, and the number of frames for each segment.
        :return: an augmented tensor of shape ``(B, T)``.
        """
        if self.token_type in ("wavlm", "hubert"):
            assert len(tokens.shape) == 2, (
                "SpecAugment only supports batches of " "single-channel token matrices."
            )
        elif self.token_type == "vq-wav2vec":
            assert len(tokens.shape) == 3
            assert tokens.shape[2] == 2
        elif self.token_type == "encodec":
            assert len(tokens.shape) == 3
            assert tokens.shape[2] == 8

        tokens = tokens.clone()
        frequency_masks = []
        for sequence_idx in range(tokens.size(0)):
            tokens[sequence_idx], frequency_mask = self._forward_single(
                tokens[sequence_idx], num_tokens, frequency_size
            )
            frequency_masks.append(frequency_mask)

        frequency_masks = torch.stack(frequency_masks, dim=0)

        return tokens, frequency_masks

    def _forward_single(
        self,
        tokens: torch.Tensor,
        num_tokens: int,
        frequency_size: int,
        warp: bool = True,
        mask: bool = True,
    ) -> torch.Tensor:
        """
        Apply SpecAugment to a single token matrix of shape (T, F).
        """
        frequency_mask = torch.zeros(frequency_size).bool()

        if random.random() > self.p:
            # Randomly choose whether this transform is applied
            return tokens, frequency_mask

        if warp:
            if self.time_warp_factor is not None and self.time_warp_factor >= 1:
                tokens = time_warp(tokens, factor=self.time_warp_factor)

        if mask:
            # Time masking
            max_tot_mask_frames = self.max_frames_mask_fraction * tokens.size(0)
            num_frame_masks = min(
                self.num_frame_masks,
                math.ceil(max_tot_mask_frames / self.frames_mask_size),
            )
            max_mask_frames = min(
                self.frames_mask_size, max_tot_mask_frames // num_frame_masks
            )
            tokens = time_mask(
                tokens,
                mask_size=max_mask_frames,
                mask_times=num_frame_masks,
                mask_value=num_tokens,
            )
            # Frequency masking
            frequency_mask = generate_frequency_mask(
                frequency_size=frequency_size,
                mask_size=self.tokens_mask_size,
                mask_times=self.num_token_masks,
            )

        return tokens, frequency_mask

    def state_dict(self, **kwargs) -> Dict[str, Any]:
        return dict(
            time_warp_factor=self.time_warp_factor,
            num_token_masks=self.num_token_masks,
            tokens_mask_size=self.tokens_mask_size,
            num_frame_masks=self.num_frame_masks,
            frames_mask_size=self.frames_mask_size,
            max_frames_mask_fraction=self.max_frames_mask_fraction,
            p=self.p,
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.time_warp_factor = state_dict.get(
            "time_warp_factor", self.time_warp_factor
        )
        self.num_token_masks = state_dict.get("num_token_masks", self.num_token_masks)
        self.tokens_mask_size = state_dict.get(
            "tokens_mask_size", self.tokens_mask_size
        )
        self.num_frame_masks = state_dict.get("num_frame_masks", self.num_frame_masks)
        self.frames_mask_size = state_dict.get(
            "frames_mask_size", self.frames_mask_size
        )
        self.max_frames_mask_fraction = state_dict.get(
            "max_frames_mask_fraction", self.max_frames_mask_fraction
        )
        self.p = state_dict.get("p", self.p)


def time_mask(
    tokens: torch.Tensor,
    mask_size: int,
    mask_times: int,
    mask_value: float,
) -> torch.Tensor:
    """
    Apply Time masking.
    Frequency and Time masking as described in the SpecAugment paper.

    :param tokens: input tensor of shape ``(T)``
    :mask_size: the width size for masking.
    :mask_times: the number of masking regions.
    :mask_value: Value to assign to the masked regions.
    """
    values = torch.randint(int(0), int(mask_size), (1, mask_times))
    min_values = torch.rand(1, mask_times) * (tokens.size(0) - values)
    mask_starts = (min_values.long()).squeeze()
    mask_ends = (min_values.long() + values.long()).squeeze()

    if mask_times == 1:
        tokens[mask_starts:mask_ends] = mask_value
    else:
        for mask_start, mask_end in zip(mask_starts, mask_ends):
            tokens[mask_start:mask_end] = mask_value

    return tokens


def generate_frequency_mask(
    frequency_size: int,
    mask_size: int,
    mask_times: int,
) -> torch.Tensor:
    """
    Generate Frequency mask.
    Frequency and Time masking as described in the SpecAugment paper.

    :frequence_size: the size of frequency.
    :mask_size: the width size for masking.
    :mask_times: the number of masking regions.
    """
    values = torch.randint(int(0), int(mask_size), (1, mask_times))
    min_values = torch.rand(1, mask_times) * (frequency_size - values)
    mask_starts = (min_values.long()).squeeze()
    mask_ends = (min_values.long() + values.long()).squeeze()

    frequency_mask = torch.zeros(frequency_size).bool()
    if mask_times == 1:
        frequency_mask[mask_starts:mask_ends] = True
    else:
        for mask_start, mask_end in zip(mask_starts, mask_ends):
            frequency_mask[mask_start:mask_end] = True

    return frequency_mask


def time_warp(tokens: torch.Tensor, factor: int) -> torch.Tensor:
    """
    :param tokens: input tensor of shape ``(T)``
    :param factor: time warping parameter.
    :return: a warped tensor of shape ``(T)``
    """
    t = tokens.size(0)
    if t - factor <= factor + 1:
        return tokens
    center = np.random.randint(factor + 1, t - factor)
    warped = np.random.randint(center - factor, center + factor + 1)
    if warped == center:
        return tokens
    tokens = tokens.unsqueeze(0).unsqueeze(0).to(torch.float32)
    left = torch.nn.functional.interpolate(
        tokens[:, :, :center],
        size=warped if len(tokens.shape) == 3 else (warped, tokens.shape[3]),
        mode="nearest",
    )
    right = torch.nn.functional.interpolate(
        tokens[:, :, center:],
        size=t - warped if len(tokens.shape) == 3 else (t - warped, tokens.shape[3]),
        mode="nearest",
    )
    return torch.cat((left, right), dim=2).squeeze(0).squeeze(0).to(torch.int64)