#!/usr/bin/env python3
# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
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

import copy
import math
import warnings
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from encoder_interface import EncoderInterface
from scaling import (
    Balancer,
    BiasNorm,
    Dropout2,
    ChunkCausalDepthwiseConv1d,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    convert_num_channels,
)
from torch import Tensor, nn


class Zipformer2(EncoderInterface):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.

        output_downsampling_factor (int): how much to downsample at the output.  Note:
            we also downsample by a factor of 2 in the Conv2dSubsampling encoder.
            You should probably leave this at 2.
        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        encoder_unmasked_dim (int or Tuple[int]): unmasked dimension in each of
            the encoder stacks for purposes of per-frame dropout (recommend 256 for
            now).
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.  Must not be less than cnn_module_kernel (after factoring in
           rounding and downsampling); an error will be thrown if this is violated.
        memory_dim: if supplied and >0, will be the dimension of the memory embeddings
            passed into the zipformer (e.g. this might be the output of another
            Zipformer used to create embedding vectors.)
    """
    def __init__(
            self,
            output_downsampling_factor: int = 2,
            downsampling_factor: Tuple[int] = (2, 4),
            encoder_dim: Union[int, Tuple[int]] = 384,
            num_encoder_layers: Union[int, Tuple[int]] = 4,
            encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
            query_head_dim: Union[int, Tuple[int]]  = 24,
            pos_head_dim: Union[int, Tuple[int]]  = 4,
            value_head_dim: Union[int, Tuple[int]] = 12,
            num_heads: Union[int, Tuple[int]] = 8,
            feedforward_dim: Union[int, Tuple[int]] = 1536,
            cnn_module_kernel: Union[int, Tuple[int]] = 31,
            memory_dim: int = -1,
            pos_dim: int = 192,
            dropout: FloatLike = None,  # see code below for default
            warmup_batches: float = 4000.0,
            causal: bool = False,
            chunk_size: Tuple[int] = (-1,),
            left_context_frames: Tuple[int] = (-1,),
    ) -> None:
        super(Zipformer2, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3),
                                     (20000.0, 0.1))

        def _to_tuple(x):
            """ Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor # int
        self.downsampling_factor = downsampling_factor # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim) # tuple
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(encoder_unmasked_dim) # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        query_head_dim = _to_tuple(query_head_dim)
        value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u,d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):

            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                memory_dim=memory_dim,
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    downsample=downsampling_factor[i],
                    dropout=dropout,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        self.downsample_output = SimpleDownsample(max(encoder_dim),
                                                  downsample=output_downsampling_factor,
                                                  dropout=dropout)

    def get_feature_masks(
            self,
            x: torch.Tensor) -> List[Union[float, Tensor]]:
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all enocder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoer dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_subsampling_factor times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (1, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dim)
        if not self.training:
            return [ 1.0 ] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0

        feature_mask_dropout_prob = 0.125

        # mask1 shape: (1, batch_size, 1)
        mask1 = (torch.rand(1, batch_size, 1,
                            device=x.device) >
                 feature_mask_dropout_prob).to(x.dtype)

        # mask2 has additional sequences masked, about twice the number.
        mask2 = torch.logical_and(mask1,
                                  (torch.rand(1, batch_size, 1,
                                              device=x.device) >
                                   feature_mask_dropout_prob).to(x.dtype))


        # dim: (1, batch_size, 2)
        mask = torch.cat((mask1, mask2), dim=-1)

        feature_masks = []
        for i in range(num_encoders):
            channels = self.encoder_dim[i]
            feature_mask = torch.ones(1, batch_size, channels,
                                       dtype=x.dtype, device=x.device)
            u1 = self.encoder_unmasked_dim[i]
            u2 = u1 + (channels - u1) // 2

            feature_mask[:, :, u1:u2] *= mask[..., 0:1]
            feature_mask[:, :, u2:] *= mask[..., 1:2]

            feature_masks.append(feature_mask)

        return feature_masks


    def get_chunk_info(self) -> Tuple[int, int]:
        """
        Returns chunk_size and left_context_chunks.
        """
        if not self.causal:
            return -1, -1
        chunk_size = random.choice(self.chunk_size)
        if chunk_size == -1:
            left_context_chunks = -1
        else:
            left_context_frames = random.choice(self.left_context_frames)
            # Note: in Python, -1 // n == -1 for n > 0
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1
        return chunk_size, left_context_chunks


    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
             masked position. May be None.
          memory:  optionally, the memory embeddings of shape (memory_len, batch_size, memory_dim)
          memory_key_padding_mask: optionally the mask for padding of memory input (for source-
             attention), of shape  (batch_size, memory_len); True means
              masked position.  May be None.

        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        outputs = []
        feature_masks = self.get_feature_masks(x)

        chunk_size, left_context_chunks = self.get_chunk_info()

        attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        if self.training and memory is not None:
            batch_size = x.shape[1]
            # setting memory to zero should be equivalent to not using the
            # memory input at all, since the Attention module has no biases.
            memory_dropout_rate = 0.05
            memory = memory * (torch.rand(batch_size, 1, device=memory.device) >
                               memory_dropout_rate)

        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])

            x = module(x,
                       chunk_size=chunk_size,
                       feature_mask=feature_masks[i],
                       src_key_padding_mask=(None if src_key_padding_mask is None
                                             else src_key_padding_mask[...,::ds]),
                       attn_mask=attn_mask,
                       memory=memory,
                       memory_key_padding_mask=memory_key_padding_mask,
            )
            outputs.append(x)

        def get_full_dim_output():
            num_encoders = len(self.encoder_dim)
            assert len(outputs) == num_encoders
            output_dim = max(self.encoder_dim)
            output_pieces = [ outputs[-1] ]
            cur_dim = self.encoder_dim[-1]
            for i in range(num_encoders - 2, -1, -1):
                d = self.encoder_dim[i]
                if d > cur_dim:
                    this_output = outputs[i]
                    output_pieces.append(this_output[..., cur_dim:d])
                    cur_dim = d
            assert cur_dim == output_dim
            return torch.cat(output_pieces, dim=-1)

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = get_full_dim_output()
        x = self.downsample_output(x)

        d = self.output_downsampling_factor
        lengths = (x_lens + d - 1) // d

        return x, lengths

    def _get_attn_mask(self, x: Tensor,
                       chunk_size: int,
                       left_context_chunks: int
    ) -> Optional[Tensor]:
        """
        Return None if chunk_size == -1, else return attention mask of shape
          (seq_len, seq_len), interpreted as (tgt_seq_len, src_seq_len).  True
           means a masked position.
        Args:
           x: embeddings after self.encoder_embed(), of shape (seq_len, batch_size, embed_dim).
          chunk_size: chunk size, must divide
        """
        if chunk_size <= 0:
            return None
        assert all(chunk_size % d == 0 for d in self.downsampling_factor)
        if left_context_chunks >= 0:
            num_encoders = len(self.encoder_dim)
            assert all (chunk_size * left_context_chunks >=
                        (self.cnn_module_kernel[i] // 2) * self.downsampling_factor[i]
                        for i in range(num_encoders))
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        # c is chunk index for each frame, shape (seq_len,)
        c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(src_c > tgt_c,
                                     src_c < tgt_c - left_context_chunks)
        if __name__ == "__main__":
            logging.info(f"attn_mask = {attn_mask}")
        return attn_mask


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x),
                          (20000.0, ratio * x),
                          default=x)

def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))



class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """
    def __init__(
            self,
            embed_dim: int,
            pos_dim: int,
            num_heads: int,
            query_head_dim: int,
            pos_head_dim: int,
            value_head_dim: int,
            feedforward_dim: int,
            dropout: FloatLike = 0.1,
            cnn_module_kernel: int = 31,
            causal: bool = False,
            memory_dim: int = -1,
            attention_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),
            ff2_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
            ff3_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
            bypass_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.02), default=0),
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass; see its default values.
        self.bypass = BypassModule(embed_dim, skip_rate=bypass_skip_rate,
                                   straight_through_rate=0.025)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0.025)


        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, pos_dim=pos_dim, num_heads=num_heads,
            query_head_dim=query_head_dim, pos_head_dim=pos_head_dim,
            dropout=0.0,
        )


        self.self_attn1 = Attention(embed_dim, embed_dim, num_heads,
                                        value_head_dim)

        self.self_attn2 = Attention(embed_dim, embed_dim, num_heads,
                                    value_head_dim)

        if memory_dim > 0:
            self.attn_weights = MultiheadAttentionWeights(
                memory_dim, embed_dim,
                num_heads=num_heads,
                head_dim=query_head_dim,
                dropout=0.0,
            )
            self.src_attn1 = Attention(memory_dim, embed_dim, num_heads,
                                       value_head_dim)
            self.src_attn2 = Attention(memory_dim, embed_dim, num_heads,
                                       value_head_dim)


        self.feed_forward1 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 3) // 4,
                                               dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim,
                                               feedforward_dim,
                                               dropout)

        feed_forward3a = FeedforwardModule(embed_dim,
                                           (feedforward_dim * 5) // 4,
                                           dropout=0.0)
        feed_forward3b = FeedforwardModule(embed_dim,
                                           (feedforward_dim * 5) // 4,
                                           dropout=0.0)
        self.feed_forward3 = ChoiceModule(embed_dim,
                                          feed_forward3a,
                                          feed_forward3b)

        self.nonlin_attention = NonlinAttention(embed_dim,
                                                hidden_channels=3 * embed_dim // 4)

        self.conv_module1 = ConvolutionModule(embed_dim,
                                             cnn_module_kernel,
                                             causal=causal)

        self.conv_module2 = ConvolutionModule(embed_dim,
                                              cnn_module_kernel,
                                              causal=causal)


        #self.attention_squeeze = AttentionSqueeze(embed_dim, embed_dim // 2)

        self.norm = BiasNorm(embed_dim)

        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.balancer1 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.2, max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule
        self.balancer_na = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of feedforward2, prevent it from staying too
        # small.  give this a very small probability, even at the start of
        # training, it's to fix a rare problem and it's OK to fix it slowly.
        self.balancer_ff2 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.balancer_ff3 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.2), default=0.0),
            max_abs=4.0,
            prob=0.05,
        )

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(4.0, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

        self.balancer2 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.1, max_abs=4.0,
        )


    def get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale,
                                    min=float(self.bypass_min),
                                    max=float(self.bypass_max))
            layer_skip_rate = float(self.layer_skip_rate)
            if layer_skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > layer_skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            return ans

    def get_sequence_dropout_mask(self, x: Tensor, dropout_rate: float) -> Optional[Tensor]:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting():
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask


    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask


    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
         pos_emb: (1, 2*seq_len-1, pos_emb_dim) or (batch_size, 2*seq_len-1, pos_emb_dim)
         chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
       feature_mask: something that broadcasts with src, that we'll multiply `src`
              by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
         attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
               True means masked position. May be None.
    src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
             masked position.  May be None.

        Returns:
           A tensor which has the same shape as src
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        attention_skip_rate = float(self.attention_skip_rate) if self.training else 0.0

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        if memory is not None and hasattr(self, 'attn_weights'):
            src_attn_weights = self.attn_weights(memory, src, memory_key_padding_mask)

        src = src + self.feed_forward1(src)

        attn_dropout_mask = self.get_sequence_dropout_mask(src, attention_skip_rate)

        if True:
            selected_attn_weights = attn_weights[0:2]
            if random.random() < float(self.const_attention_rate):
                # Make attention weights constant.  The intention is to
                # encourage these modules to do something similar to an
                # averaging-over-time operation.
                # only need the mask, can just use the 1st one and expand later
                selected_attn_weights = selected_attn_weights[0:1]
                selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
                selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))
                selected_attn_weights = selected_attn_weights.expand(2, -1, -1, -1)


        na = self.balancer_na(self.nonlin_attention(src,
                                                    selected_attn_weights[0:1]))

        src = src + (na if attn_dropout_mask is None else na * attn_dropout_mask)

        self_attn = self.self_attn1(
            src, attn_weights)

        src = src + (self_attn if attn_dropout_mask is None else self_attn * attn_dropout_mask)

        if memory is not None and hasattr(self, 'attn_weights'):
            src = src + self.sequence_dropout(self.src_attn1(memory, src_attn_weights),
                                              attention_skip_rate)

        src = src + self.sequence_dropout(self.conv_module1(src, chunk_size=chunk_size,
                                                            src_key_padding_mask=src_key_padding_mask),
                                          float(self.conv_skip_rate))

        src = src + self.sequence_dropout(self.balancer_ff2(self.feed_forward2(src)),
                                          float(self.ff2_skip_rate))

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(
            src, attn_weights)

        src = src + (self_attn if attn_dropout_mask is None else self_attn * attn_dropout_mask)

        if memory is not None and hasattr(self, 'attn_weights'):
            src = src + self.sequence_dropout(self.src_attn2(memory, src_attn_weights),
                                              attention_skip_rate)

        src = src + self.sequence_dropout(self.conv_module2(src, chunk_size=chunk_size,
                                                            src_key_padding_mask=src_key_padding_mask),
                                          float(self.conv_skip_rate))

        src = src + self.sequence_dropout(self.balancer_ff3(self.feed_forward3(src)),
                                          float(self.ff3_skip_rate))

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src

class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers

    Args:
     encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
           pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = Zipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """
    def __init__(
            self,
            encoder_layer: nn.Module,
            num_layers: int,
            pos_dim: int,
            dropout: float,
            warmup_begin: float,
            warmup_end: float,
            initial_layerdrop_rate: float = 0.5,
            final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, dropout_rate=0.15,
                                                        length_factor=1.0)

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end

        delta = (1. / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.layers[i].bypass.skip_rate = ScheduledFloat((cur_begin, initial_layerdrop_rate),
                                                             (cur_end, final_layerdrop_rate),
                                                             default=0.0)
            cur_begin = cur_end

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.
            memory:  optionally, the memory embeddings of shape (memory_len, batch_size, memory_dim)
            memory_key_padding_mask: optionally the mask for padding of memory input (for source-
                attention), of shape  (batch_size, memory_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        pos_emb = self.encoder_pos(src)
        output = src

        rnd_seed = src.numel() + random.randint(0, 1000)

        output = output * feature_mask

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            output = output * feature_mask

        return output


class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """
    def __init__(
            self,
            embed_dim: int,
            skip_rate: FloatLike = 0.0,
            straight_through_rate: FloatLike = 0.0,
            scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
            scale_max: FloatLike = 1.0):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)


    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale,
                                    min=float(self.scale_min),
                                    max=float(self.scale_max))
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) < straight_through_rate
                ans = torch.maximum(ans, mask.to(ans.dtype))

            return ans

    def forward(self,
                src_orig: Tensor,
                src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig)  * bypass_scale




class DownsampledZipformer2Encoder(nn.Module):
    r"""
    DownsampledZipformer2Encoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """
    def __init__(self,
                 encoder: nn.Module,
                 dim: int,
                 downsample: int,
                 dropout: FloatLike):
        super(DownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim,
                                           downsample, dropout)
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = BypassModule(dim, straight_through_rate=0.025)


    def forward(self,
                src: Tensor,
                chunk_size: int = -1,
                feature_mask: Union[Tensor, float] = 1.0,
                attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.
            memory:  optionally, the memory embeddings of shape (memory_len, batch_size, memory_dim)
            memory_key_padding_mask: optionally the mask for padding of memory input (for source-
                attention), of shape  (batch_size, memory_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if attn_mask is not None:
            attn_mask = attn_mask[::ds,::ds]

        src = self.encoder(
            src,
            chunk_size=chunk_size // ds,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[:src_orig.shape[0]]

        return self.out_combiner(src_orig, src)



class SimpleDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """
    def __init__(self,
                 channels: int,
                 downsample: int,
                 dropout: FloatLike):
        super(SimpleDownsample, self).__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None # will be set from training code
        self.dropout = copy.deepcopy(dropout)

        self.downsample = downsample

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        if seq_len != d_seq_len * ds:
            # right-pad src, repeating the last element.
            pad = d_seq_len * ds - seq_len
            src_extra = src[src.shape[0]-1:].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)
            assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        # weights: (downsample, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)

        return ans


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """
    def __init__(self,
                 num_channels: int,
                 upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works done by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish invidual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embedding_dim)


    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """
    def __init__(
        self, embed_dim: int,
            dropout_rate: FloatLike,
            max_len: int = 1000,
            length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))



    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(0) >= x.size(0) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(
                    x.device
                ):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        T = x.size(0)
        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(T-1), T,
                         device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = (self.embed_dim ** 0.5)
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = compression_length * x.sign() * ((x.abs() + compression_length).log() - math.log(compression_length))

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan() # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.to(dtype=x.dtype)


    def forward(self, x: torch.Tensor) -> Tensor:
        """Create positional encoding.

        Args:
            x (torch.Tensor): Input tensor (time, batch, `*`).

        Returns:
            positional embedding, of shape (1, 2*time-1, `*`).

        """
        self.extend_pe(x)
        pos_emb = self.pe[
            self.pe.size(0) // 2
            - x.size(0)
            + 1 : self.pe.size(0) // 2  # noqa E203
            + x.size(0),
            :
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)



class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(
            self,
            embed_dim: int,
            pos_dim: int,
            num_heads: int,
            query_head_dim: int,
            pos_head_dim: int,
            dropout: float = 0.0,
            pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5),
                                                          (4000.0, 0.0))
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(embed_dim, in_proj_dim, bias=True,
                                    initial_scale=query_head_dim**-0.25)

        self.whiten_keys = Whiten(num_groups=num_heads,
                                  whitening_limit=_whitening_schedule(3.0),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.025)

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be suffixient to fix the problem.
        self.balance_keys = Balancer(key_head_dim * num_heads,
                                     channel_dim=-1,
                                     min_positive=0.4,
                                     max_positive=0.6,
                                     min_abs=0.0,
                                     max_abs=100.0,
                                     prob=0.025)


        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(pos_dim,
                                       num_heads * pos_head_dim,
                                       bias=False,
                                       initial_scale=0.05)


        # the following are for diagnosics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()


    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 2, pos_dim)
           chunk_size
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len),
               interpreted as ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        q = x[...,0:query_dim]
        k = x[...,query_dim:2*query_dim]
        # p is the position-encoding query
        p = x[...,2*query_dim:]
        assert p.shape[-1] == num_heads * pos_head_dim


        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.


        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        if not self.training or random.random() >= float(self.pos_emb_skip_rate):
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            pos_scores = pos_scores.as_strided((num_heads, batch_size, seq_len, seq_len),
                                               (pos_scores.stride(0),
                                                pos_scores.stride(1),
                                                pos_scores.stride(2)-pos_scores.stride(3),
                                                pos_scores.stride(3)),
                                               storage_offset=pos_scores.stride(3) * (seq_len - 1))

            attn_scores = attn_scores + pos_scores

        if self.training and random.random() < 0.1:
            # This is away of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 25.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(attn_scores,
                                                 limit=25.0,
                                                 penalty=1.0e-04,
                                                 name=self.name)

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, seq_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if random.random() < 0.001:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights


    def _print_attn_entropy(
            self,
            attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(
                    dim=-1).mean(dim=(1,2))
                logging.info(f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}")


class Attention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim_in: the input embedding dimension
          embed_dim_out: the output embedding dimension (normally the same as input)
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """
    def __init__(
            self,
            embed_dim_in: int,
            embed_dim_out: int,
            num_heads: int,
            value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim_in,
                                 num_heads * value_head_dim,
                                 bias=False)

        self.out_proj = ScaledLinear(num_heads * value_head_dim,
                                     embed_dim_out, bias=False,
                                     initial_scale=0.05)

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)


    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, query_len, key_len),
          Expect attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (num_heads, batch_size, query_len, key_len) = attn_weights.shape

        x = self.in_proj(x)     #  (key_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(key_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, key_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, query_len, value_head_dim)

        x = x.permute(2, 1, 0, 3).contiguous().view(
            query_len, batch_size, num_heads * value_head_dim)

        # returned value is of shape (query_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)

        return x


class MultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head cross-attention weights.  Allows src and target
    to have different dims.

    Args:
          key_embed_dim: number of channels of the thing that we'll project to
              make the query (corresponds to source).  e.g. 256
          query_embed_dim: number of channels of the thing that we'll project to
              make the query (corresponds to target).  e.g. 256
          num_heads:  number of heads to compute weights for, e.g. 8
           head_dim: dimension of the query and key, per head.  e.g. 24.
             dropout: dropout probability for attn_output_weights. Default: 0.0.
    """

    def __init__(
            self,
            key_embed_dim: int,
            query_embed_dim: int,
            num_heads: int,
            head_dim: int,
            dropout: float = 0.0,

    ) -> None:
        super().__init__()
        self.key_embed_dim = key_embed_dim
        self.query_embed_dim = query_embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.name = None  # will be overwritten in training code; for diagnostics.


        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.query_in_proj = ScaledLinear(query_embed_dim,
                                          head_dim * num_heads,
                                          bias=True,
                                          initial_scale=head_dim ** -0.25)

        # weights produced by this module are invariant to adding a constant to
        # the keys, so we don't need a bias for the keys.
        self.key_in_proj = ScaledLinear(key_embed_dim,
                                        head_dim * num_heads,
                                        bias=False,
                                        initial_scale=head_dim ** -0.25)

        self.whiten_keys = Whiten(num_groups=num_heads,
                                  whitening_limit=_whitening_schedule(3.0),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.025)



    def forward(
        self,
        key: Tensor,
        query: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
              key: input of shape (key_len, batch_size, key_embed_dim)
            query: input of shape (query_len, batch_size, query_embed_dim)
          key_padding_mask: an optional bool tensor of shape (batch_size, key_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, query_len, key_len)
        """
        q = self.query_in_proj(query)
        k = self.key_in_proj(key)

        head_dim = self.head_dim
        num_heads = self.num_heads

        query_len, batch_size, _ = q.shape
        key_len, _batch_size, _ = k.shape
        assert _batch_size == batch_size

        k = self.whiten_keys(k)   # does nothing in the forward pass.

        q = q.reshape(query_len, batch_size, num_heads, head_dim)
        k = k.reshape(key_len, batch_size, num_heads, head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        if self.training and random.random() < 0.1:
            # This is a way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 25.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(attn_scores,
                                                 limit=25.0,
                                                 penalty=1.0e-04,
                                                 name=self.name)

        assert attn_scores.shape == (num_heads, batch_size, query_len, key_len)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, key_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if random.random() < 0.001:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights


    def _print_attn_entropy(
            self,
            attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(
                    dim=-1).mean(dim=(1,2))
                logging.info(f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}")


class ChoiceModule(nn.Module):
    """
    This module multiplexes frames between two submodules that are passed into it, in a
    learnable way.  The submodules passed into it should have the same number of channels in
    their input and output, and should not have any sequence-wise operation, i.e. they
    should operated independently per "frame" where a "frame" is just a vector of activations with
    `num_channels` channels.

    The idea is that you might have two versions of a feedforward module, for instance, and
    allocate about half of the frames to each verison; this would allow you in principle to
    use double the parameters without using very much more memory or time for training.
    """
    def __init__(self,
                 num_channels: int,
                 module1: nn.Module,
                 module2: nn.Module,
                 min_proportion2: FloatLike = 0.2,
                 max_proportion2: FloatLike = 0.8,
                 intermediate_rate: FloatLike = 0.05):
        super().__init__()
        # the min_abs and max_abs constraints are very arbitrary just to keep it in
        # a consistent range for model averaging, since it's only going to be the
        # relative values of the scores that matter.
        self.score_balancer = Balancer(1,
                                       channel_dim=-1,
                                       min_positive=min_proportion2,
                                       max_positive=max_proportion2,
                                       min_abs=0.8,
                                       max_abs=1.2,
                                       prob=0.5)

        # self.params is the projection to the scores.
        self.to_scores = nn.Linear(num_channels, 1, bias=False)

        # intermediate_rate is the target proportion of the activations that we
        # aim to have intermediate weights between 0 and 1; this requires that
        # we evaluate both sub-networks for them, and we don't want this to be
        # too many or it would be slow; but we also can't let it be zero or we
        # wouldn't be able to train the score projection.
        self.intermediate_rate = copy.deepcopy(intermediate_rate)

        self.module1 = module1
        self.module2 = module2

    def forward(self,
                x: Tensor):
        """
        Forward function.
          x: a Tensor of shape (*, num_channels)
        Returns:
         a Tensor with the same shape as x.
        """
        x_shape = x.shape
        num_channels = x_shape[-1]
        x = x.reshape(-1, num_channels)

        scores = self.to_scores(x)  # (num_frames, 1)

        scores = self.score_balancer(scores)

        scores = scores.flatten() # (num_frames,)

        module1_indexes, module2_indexes, module2_weight, reverse_indexes = self.process_scores(scores)

        self._test_indexes(module1_indexes, module2_indexes, module2_weight, reverse_indexes)

        x1 = torch.index_select(x, dim=0, index=module1_indexes)
        x2 = torch.index_select(x, dim=0, index=module2_indexes)

        x1 = self.module1(x1)
        x2 = self.module2(x2)

        num_overlapping = module2_weight.numel()
        if num_overlapping > 0:
            module1_weight = 1.0 - module2_weight
            x_overlapping = (x1[-num_overlapping:] * module1_weight.unsqueeze(-1) +
                             x2[:num_overlapping] * module2_weight.unsqueeze(-1))
            x = torch.cat((x1[:-num_overlapping],
                           x_overlapping,
                           x2[num_overlapping:]),
                          dim=0)
        else:
            x = torch.cat((x1, x2),
                          dim=0)

        x = torch.index_select(x, dim=0, index=reverse_indexes)
        x = x.reshape(x_shape)
        return x


    def choose_modules(self, x: Tensor, module1: nn.Module, module2: nn.Module,
                       module1_indexes: Tensor, module2_indexes: Tensor,
                       module2_weight: Tensor, reverse_indexes: Tensor):
        """
        Pass x through two modules according to provided indexes and possibly weights.
        Args:
                       x: the input Tensor, of shape (num_frames, num_channels)
          module1_indexes: the subset of frame indexes 0..num_frames-1 that should be passed through
                         module1
          module2_indexes: the subset of frame indexes 0..num_frames-1 that should be passed through
                         module2.  In training mode there may be some overlap between the two; the
                         overlapping elements should be at the end of module1 and the beginning of module2.
        """



    def _test_indexes(self, module1_indexes, module2_indexes,
                      module2_weight, reverse_indexes):
        if __name__ != '__main__' and random.random() > 0.025:
            return
        num_frames = reverse_indexes.numel()
        num_overlap = module2_weight.numel()
        assert module1_indexes.numel() + module2_indexes.numel() - num_overlap == num_frames
        if num_overlap > 0:
            assert torch.all(module1_indexes[-num_overlap:] == module2_indexes[:num_overlap])
        forward_indexes = torch.cat((module1_indexes,
                                     module2_indexes[num_overlap:]))
        # make sure that forward_indexes and reverse_indexes are inverse permutations.
        assert torch.all(reverse_indexes[forward_indexes] == torch.arange(num_frames,
                                                                          device=forward_indexes.device))
        if not (torch.all(module2_weight >= 0) and torch.all(module2_weight <= 1)):
            logging.info(f"module2_weight={module2_weight}, min={module2_weight.min()}, max={module2_weight.max()}")
        if module2_weight.numel() > 0:
            assert module2_weight[-1] >= module2_weight[0]



    def process_scores(self,
                       scores: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Process a tensor of `scores` into 3 tensors that indicate which frames to
        evaluate for the two sub-modules; and for those that are evaluated for both
        sub-modules, the weights to use for module 2 (subtract this from 1.0
        to get the weights for module 1).

        Args:
           score: a Tensor of shape (num_frames,), where negative values0 are supposed to
               indicate "use module 1" and positive values indicate "use module 2".  The
               average-absolute-value of these scores is expected to be
         Returns:
             (module1_indexes, module2_indexes, module2_weight, reverse_indexes)

           module1_indexes is a Tensor of frame-indexes (equivalent to indexes into `scores`)
                of shape (num_indexes1,) with 0 <= num_indexes1 <= num_frames, indicating which
                frames must be evaluated with module 1.
           module2_indexes is a Tensor of frame-indexes (equivalent to indexes into `scores`)
                of shape (num_indexes2,) with 0 <= num_indexes2 <= num_frames, indicating which
                frames must be evaluated with module 2.
           module2_weight is possibly-empty Tensor with elements 0 <= module2_weight <= 1
                giving the weights for module2 for the small proportion of frames that have
                intermediate weights (needed during training to train the scores).  In
                test-time this will be an empty Tensor.
           reverse_indexes is a Tensor containing a permutation of the range 0..num_frames-1,
                that is to be used when combining the outputs of the two modules back

        """
        num_frames = scores.numel()

        sscores, indexes = scores.sort()

        if self.training:
            intermediate_rate = float(self.intermediate_rate)
            collar = (0.5 * intermediate_rate)

            # intermediate_sindexes is the indexes within the sorted scores ("s"
            # for sorted) of the frames that have scores close to zero (within
            # "collar").
            intermediate_sindexes, = (sscores.abs() < collar).nonzero(as_tuple=True)

        def get_reverse_indexes():
            # num_intermediate is the number of elements in module1_indexes and module2_indexes
            # that are the same (these indentical elements will be at the end of
            # module1_indexes and the beginning of module2_indexes
            num_intermediate = module2_weight.numel()
            forward_indexes = torch.cat((module1_indexes, module2_indexes[num_intermediate:]))
            # invert the permutation `forward_indexes`
            arange = torch.arange(num_frames, device=forward_indexes.device)
            reverse_indexes = torch.empty(num_frames, dtype=torch.long,
                                          device=forward_indexes.device)
            reverse_indexes.scatter_(dim=0, index=forward_indexes, src=arange)
            return reverse_indexes


        if not self.training or intermediate_sindexes.numel() == 0:
            module1_indexes = indexes[(sscores < 0).nonzero(as_tuple=True)[0]]
            module2_indexes = indexes[(sscores >= 0).nonzero(as_tuple=True)[0]]
            module2_weight = torch.zeros(0, device=scores.device)
            return module1_indexes, module2_indexes, module2_weight, get_reverse_indexes()


        # max_intermediate is to prevent exhausting memory.
        max_intermediate = int(2 * intermediate_rate * num_frames) + 10

        # the number of intermediate_sindexes is not huge, so we won't run into memory
        # problems if we evaluate both modules with this many indexes.
        intermediate_sindexes = intermediate_sindexes.to('cpu')

        first_in_collar = intermediate_sindexes[0]
        last_in_collar = intermediate_sindexes[-1]

        if last_in_collar - first_in_collar > max_intermediate:
            n = ((last_in_collar - first_in_collar) - max_intermediate) // 2
            first_in_collar += n
            last_in_collar -= n
            # note, this way of doing it is not super-ideal, but the balancer
            # should prevent this kind of thing from happening except
            # at at the early stages of training.

        # module1_indexes includes those to the left of the collar and
        # inside the collar
        module1_indexes = indexes[:last_in_collar+1]
        # module2_indexes includes those inside the collar and those to the
        # right of the collar.
        module2_indexes = indexes[first_in_collar:]

        module2_weight = sscores[first_in_collar:last_in_collar+1]

        # note, collar == 0.5 * intermediate_rate.
        module2_weight = (module2_weight + collar) * (1.0 / intermediate_rate)

        return module1_indexes, module2_indexes, module2_weight, get_reverse_indexes()


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model.
    """
    def __init__(self,
                 embed_dim: int,
                 feedforward_dim: int,
                 dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(feedforward_dim,
                                        channel_dim=-1,
                                        min_positive=0.3,
                                        max_positive=1.0,
                                        min_abs=0.75,
                                        max_abs=5.0)

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(feedforward_dim, embed_dim,
                                                   activation='SwooshL',
                                                   dropout_p=dropout,
                                                   dropout_shared_dim=0, bias=True,
                                                   initial_scale=0.1)

        self.out_whiten =  Whiten(num_groups=1,
                                  whitening_limit=_whitening_schedule(7.5),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.01)

    def forward(self,
                x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
            self,
            channels: int,
            hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at 2.0,
        # because we noticed that well-trained instances of this module have abs-value before the sigmoid
        # starting from about 3, and poorly-trained instances of the module have smaller abs values
        # before the sigmoid.
        self.balancer = Balancer(
            hidden_channels, channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(hidden_channels, channels,
                                     bias=True,
                                     initial_scale=0.05)



        self.whiten1 = Whiten(num_groups=1,
                              whitening_limit=_whitening_schedule(5.0),
                              prob=(0.025, 0.25),
                              grad_scale=0.01)

        self.whiten2 = Whiten(num_groups=1,
                              whitening_limit=_whitening_schedule(5.0, ratio=3.0),
                              prob=(0.025, 0.25),
                              grad_scale=0.01)


    def forward(self,
                x: Tensor,
                attn_weights: Tensor,
    ) -> Tensor:
        """.
        Args:
           x: a Tensor of shape (seq_len, batch_size, num_channels)
attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
        Returns:
           a Tensor with the same shape as x
        """
        num_channels = x.shape[-1]
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=-1)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)


        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """
    def __init__(
            self, channels: int, kernel_size: int, causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels, 2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        # after in_proj we put x through a gated linear unit (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in the range 1 to 4,
        # but sometimes, for some reason, for layer 0 the rms ends up being very large,
        # between 50 and 100 for different channels.  This will cause very peaky and
        # sparse derivatives for the sigmoid gating function, which will tend to make
        # the loss function not learn effectively.  (for most layers the average absolute values
        # are in the range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for different
        # layers, which likely breaks down as 0.5 for the "linear" half and
        # 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that if we
        # constrain the rms values to a reasonable range via a constraint of max_abs=10.0,
        # it will be in a better position to start learning something, i.e. to latch onto
        # the correct range.
        self.balancer1 = Balancer(
            bottleneck_dim, channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity() # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity() # for diagnostics

        assert kernel_size % 2 == 1

        self.depthwise_conv = ChunkCausalDepthwiseConv1d(
            channels=bottleneck_dim,
            kernel_size=kernel_size) if causal else nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            groups=bottleneck_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2)

        self.balancer2 = Balancer(
            bottleneck_dim, channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim, channels, activation='SwooshR',
            dropout_p=0.0, initial_scale=0.05,
        )

    def forward(self,
                x: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                chunk_size: int = -1,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=-1)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        if chunk_size >= 0:
            assert self.causal, "Must initialize model with causal=True if you use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(2, 0, 1) # (time, batch, channels)

        x = self.whiten(x)  # (time, batch, channels)
        x = self.out_proj(x)  # (time, batch, channels)

        return x


class ScalarMultiply(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def _test_zipformer_main(causal: bool = False):
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    memory_dim = 100

    c = Zipformer2(
        encoder_dim=(64, 96), encoder_unmasked_dim=(48, 64), num_heads=(4, 4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,),
        memory_dim=memory_dim,
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        memory=torch.randn(101, batch_size, memory_dim),
    )
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_zipformer_main(False)
    _test_zipformer_main(True)
