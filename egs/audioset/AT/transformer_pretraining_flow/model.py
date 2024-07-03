from functools import partial
import logging
from typing import List, Optional, Tuple
import random

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible, get_1d_sincos_pos_embed_from_grid
from util.patch_embed import PatchEmbed_own

from icefall.utils import AttributeDict, make_pad_mask


class AudioMAEModel(nn.Module):
    def __init__(
        self,
        encoder_dim: int=1024,
        num_encoder_layers: int=12,
        num_heads: int=16,
        mlp_ratio: int=4.0,
        decoder_dim: int=512,
        num_decoder_layers: int=8,
        num_heads_decoder: int=16,
        input_dim: int=80,
        patch_width: int=4,
        max_num_patches: int = 1500,
        mask_prob: float = 0.7,
        mask_length: int = 5,
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.input_dim = input_dim
        self.patch_width = patch_width
        
        # encoder related
        self.patch_embed = PatchEmbed_own(
            n_mels=input_dim,
            patch_width=patch_width,
            in_chans=1,
            embed_dim=encoder_dim,
            stride=patch_width,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    encoder_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer
                ) 
                for i in range(num_encoder_layers)
            ]
        )
        self.encoder_dim = encoder_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, encoder_dim)) 
        self.norm = norm_layer(encoder_dim)

        # decoder related
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_dim,
                    num_heads_decoder,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer
                ) 
                for i in range(num_decoder_layers)
            ]
        )

        self.decoder_dim = decoder_dim
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, decoder_dim))  
        self.decoder_pred = nn.Linear(decoder_dim, patch_width * input_dim, bias=True)
        self.decoder_norm = norm_layer(decoder_dim)

        # mask token embedding
        self.mask_token = nn.Parameter(torch.FloatTensor(1, 1, decoder_dim).normal_())

        # mask related
        self.mask_prob = mask_prob
        self.mask_length = mask_length

    def initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, audio: torch.Tensor):
        """
        audio: (B,T,C)
        return: (B, T//patch_width, C)
        """
        pass
    
    def forward_encoder(self, x, x_lens):
        # embed patches
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)
        x = self.patch_embed(x) # (B,L,C)
        x_lens = x_lens // 4

        # add pos embed
        L = x_lens.max()
        x = x + self.pos_embed[:, :L, :]

        padding_mask = make_pad_mask(x_lens)

        # TODO: apply mask
        # masking: length -> length * mask_ratio
        # The masked indices have value 1
        x, mask_indices = self.apply_mask_custom(
            x.clone(), 
            padding_mask=padding_mask,
        ) # (B, L, C)

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.norm(x)

        ids_restore = torch.argsort(mask_indices, dim=1, stable=True)
        ids_restore = torch.argsort(ids_restore, dim=1, stable=True)

        return x, x_lens, mask_indices, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        mask_indices: torch.Tensor,
        ids_restore: torch.Tensor
    ):
        # embed tokens
        x = self.decoder_embed(x)

        d_T = x.size(1)
        B, T = ids_restore.shape

        x_ = torch.cat([x, torch.zeros(B,(T-d_T), self.decoder_dim)], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.size(2)))
        x[mask_indices] = self.mask_token

        x = x + self.decoder_pos_embed[:, :T,]

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        pred = self.decoder_pred(x) # (B,L,C)

        return pred

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        encoder_out, encoder_out_lens, mask_indices, ids_restore = self.forward_encoder(x, x_lens)
        pred = self.forward_decoder(encoder_out, encoder_out_lens, mask_indices, ids_restore)
        loss = self.forward_loss(pred=pred, target=x, mask_indices=mask_indices)

        return loss

    def forward_loss(self, pred: torch.Tensor, target: torch.Tensor, mask_indices: torch.Tensor):
        B,T, n_mels = target.shape 
        num_patches = T//4
        target = target[:, :4 *num_patches].reshape(B, num_patches, 4 * n_mels) # (B, L, C)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # (B, L)

        loss = (loss * mask_indices).sum() / mask.sum() # mean loss on the removed patches

        return loss

    def apply_mask_custom(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        min_mask: int = 2,
    ):
        """A simpler way of applying mask to the features. It enables
        a larger masking ratio. The masking does not allow overlap, so the mask length is
        always a multiple of mask_length (when multiple masks are continuous).

        Args:
            x (torch.Tensor): the input feature to be masked
            padding_mask (torch.Tensor): the padding mask of x, True on padding positions
            min_mask (int): minimum number of mask for each sample
        """
        B,T,C = x.shape
        assert self.mask_prob > 0.0

        mask_indices = []
        for i in range(B):
            num_segments = (T - padding_mask[i].sum()) // self.mask_length # discard the last few frames
            segment_mask = torch.rand(num_segments) < self.mask_prob 
            while sum(segment_mask) < min_mask:
                segment_mask = torch.rand(num_segments) < self.mask_prob
            segment_mask_expanded = segment_mask.unsqueeze(-1).expand(num_segments, self.mask_length)
            segment_mask_expanded = segment_mask_expanded.reshape(-1)
            if segment_mask_expanded.size(0) < T:
                pad = T - segment_mask_expanded.size(0)
                segment_mask_expanded = torch.cat([segment_mask_expanded, torch.zeros(pad)])
            mask_indices.append(segment_mask_expanded)

        mask_indices = torch.stack(mask_indices).to(x.device).bool()
        if random.random() > 0.97:
            logging.info(f"Apply own random masking. A proportion of {mask_indices.sum()/mask_indices.numel():.2f} frames are masked")
        
        kept_indices = mask_indices == False
        x_out = []
        for i in range(B):
            x_out.append(x[i, kept_indices[i]])
        
        x_out = nn.utils.rnn.pad_sequence(x_out, batch_first=True, padding_value=0.0)

        return x_out, mask_indices

if __name__=="__main__":
    model = AudioMAEModel()
    x = torch.rand(2, 200, 80)
    x_lens = torch.tensor([200, 160])
    import pdb; pdb.set_trace()
    x_out, x_out_lens = model(x, x_lens)
    print(x_out.shape)