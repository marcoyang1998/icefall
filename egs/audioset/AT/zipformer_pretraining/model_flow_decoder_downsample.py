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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from lhotse.dataset import SpecAugment

from scaling import ScheduledFloat

from icefall.utils import AttributeDict, make_pad_mask


class AudioPretrainingModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: nn.Module,
        fbank_dim: int = 80,
        encoder_dim: int = 384,
        encoder_input_dim: int = 192,
        decoder_dim: int = 384,
        decoder_input_dim: int = 192,
        mask_prob: float = 0.65,
        mask_length: int = 10,
        mask_selection: str = "static",
        mask_other: float = 0.0,
    ):
        """An audio pretraining model

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
          noise_scale:
            The scale of the gaussia noise.
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.fbank_dim = fbank_dim
        
        self.decoder = decoder
        self.decoder_input_dim = decoder_input_dim
        self.decoder_dim = decoder_dim
        
        # proj the encoder features to decoder dim
        self.encoder_out_proj = nn.Linear(
            encoder_dim, decoder_input_dim, bias=True
        )
        self.dropout = ScheduledFloat((0.0, 0.9), (3000.0, 0.1))
        # decoder embed
        self.decoder_embed = Conv2dSubsampling(fbank_dim, decoder_input_dim)
        # decoder pred
        self.decoder_pred = nn.Linear(
            decoder_dim, fbank_dim * 4, bias=True,
        )
        
        self.fbank_norm_mean = -4.149941921234131
        self.fbank_norm_std = 4.47724723815918

        # mask embeddings
        self.mask_emb = nn.Parameter(torch.FloatTensor(fbank_dim).uniform_())

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other

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
        t: float = 0.0,
        fbank_as_target: bool=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          target:
            The reconstruction target
          t:
            The t for mixing image and noise
        Returns:
          Return the binary crossentropy loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        N, T, C = x.shape

        padding_mask = make_pad_mask(x_lens)
        fbank = x.clone()
        fbank_lens = x_lens.clone()

        # apply masking to the fbank features
        x, mask_indices = self.apply_mask_facebook(
            x.clone(),
            padding_mask=padding_mask
        ) # (N,T,C), (N,T)

        x, x_lens = self.encoder_embed(x, x_lens) # (N,T,C)
        src_key_padding_mask = make_pad_mask(x_lens)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask) # (T,N,C)        

        # replace the masked encoder_out with 0.0, as we only want the info from the un-masked regions
        mask_indices = downsample_mask_indices(mask_indices)
        assert mask_indices.size(1) >= encoder_out.size(0)
        if mask_indices.size(1) > encoder_out.size(0):
            mask_indices = mask_indices[:, :encoder_out.size(0)]
        mask_indices = mask_indices.bool().T
        encoder_out[mask_indices] = 0.0
        
        # Get the time embedding
        timestamps = t * torch.ones(N, device=x.device) # currently use a fixed t
        t_embed = timestep_embedding(timesteps=timestamps, dim=encoder_out.size(2)) 
        encoder_out = encoder_out + t_embed
        encoder_out = encoder_out.permute(1,0,2) # (N,T,C)

        # we will use the normalized fbank as the target
        fbank = (fbank - self.fbank_norm_mean) / self.fbank_norm_std
        noise = torch.randn_like(fbank)
        decoder_in = (1-timestamps[:, None, None]) * noise + timestamps[:, None, None] * fbank
            
        # perform the reconstruction
        decoder_in, decoder_in_lens = self.decoder_embed(decoder_in, fbank_lens) # project to decoder_dim & downsample to the encoder_out length, (N,T,C)
        decoder_in = nn.functional.dropout(decoder_in, p=float(self.dropout), training=self.training)
        if encoder_out.size(1) < decoder_in.size(1):
            diff = decoder_in.size(1) - encoder_out.size(1) 
            encoder_out = torch.cat([encoder_out, torch.zeros(N, diff, encoder_out.size(2), device=encoder_out.device)], dim=1)

        decoder_in = decoder_in + self.encoder_out_proj(encoder_out) # add the condition embedding
        decoder_in = decoder_in.permute(1,0,2)
        decoder_padding_mask = make_pad_mask(decoder_in_lens)
        decoder_out, decoder_out_lens = self.decoder(
            x=decoder_in,
            x_lens=decoder_in_lens, 
            src_key_padding_mask=decoder_padding_mask,
            emb=encoder_out.permute(1, 0, 2),
        )

        decoder_out = self.decoder_pred(decoder_out) 
        decoder_out = decoder_out.permute(1, 0, 2) # (T, N, C) -> (N, T, C)

        fbank = fbank[:, :decoder_out.size(1)*4, :]
        fbank = fbank.reshape(decoder_out.shape)

        if fbank_as_target:
            loss = ((decoder_out - fbank)**2).sum() / fbank.size(2)
        else:
            loss = ((decoder_out - u_t)**2).sum() / fbank.size(2)

        return loss
    
    def apply_mask_facebook(
        self,
        x: torch.Tensor,
        padding_mask,
    ):
        # this function is modified from fairseq: https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/models/wav2vec/wav2vec2.py#L429
        # The masked indices have value 1
        B, T, C = x.shape

        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                mask_type=self.mask_selection,
                mask_other=self.mask_other,
                min_masks=2,
                no_overlap=False,  # False
                min_space=1,  # 1
                require_same_masks=False,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
            mask_indices = mask_indices.float()
            if random.random() > 0.97:
                logging.info(f"A proportion of {mask_indices.sum()/mask_indices.numel():.2f} frames are masked")
        else:
            mask_indices = None

        return x, mask_indices


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 8,
        layer2_channels: int = 32,
        layer3_channels: int = 128,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
        """
        assert in_channels >= 7
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.GELU(),
        )
        self.out = nn.Linear(
            layer3_channels * (((in_channels - 1) // 2 - 1) // 2), out_channels
        )
        # set learn_eps=False because out_norm is preceded by `out`, and `out`
        # itself has learned scale, so the extra degree of freedom is not
        # needed.

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x_lens = ((x_lens-1)//2 -1)//2
        return x, x_lens
def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def downsample_mask_indices(mask, pooling_type: str="avg"):
    if pooling_type == "avg":
        # Equivalent to the Conv2dSubsampling
        mask = nn.functional.avg_pool1d(mask, kernel_size=9, stride=2, padding=0)
        # Equivalent to the SimpleDownsample
        mask = nn.functional.pad(mask, (0,1), "replicate", 0)
        mask = nn.functional.avg_pool1d(mask, kernel_size=2, stride=2, padding=0)
        mask = mask > 0.5
        mask = mask.float()
    elif pooling_type == "max":
        mask = nn.functional.max_pool1d(mask, 4)

    return mask


def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + rng.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError("this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask