import logging
from typing import List, Optional, Tuple
import random

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible, get_1d_sincos_pos_embed_from_grid
from util.patch_embed import PatchEmbed_own

from icefall.utils import AttributeDict, make_pad_mask

class AudioTaggingTransformer(nn.Module):
    def __init__(
        self,
        encoder_dim: int=1024,
        num_encoder_layers: int=12,
        num_heads: int=16,
        mlp_ratio: int=4.0,
        input_dim: int=80,
        patch_width: int=4,
        max_num_patches: int = 1500,
        num_classes: int = 527,
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

        self.pred = nn.Linear(encoder_dim, num_classes, bias=True)

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

    def forward_encoder(self, x, x_lens):
        # embed patches
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)
        x = self.patch_embed(x) # (B,L,C)
        x_lens = x_lens // self.patch_width

        # add pos embed
        L = x_lens.max()
        x = x + self.pos_embed[:, :L, :]

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.norm(x)

        return x, x_lens

    def forward_audio_tagging(self, x, x_lens):
        logits = self.pred(x)
        padding_mask = make_pad_mask(x_lens)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
        return logits

    def forward(self, x, x_lens, target):
        import pdb; pdb.set_trace()
        x, x_lens = self.forward_encoder(x, x_lens)
        logits = self.forward_audio_tagging(x, x_lens)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum")

        return loss

if __name__=="__main__":
    model = AudioTaggingTransformer(
        encoder_dim=512,
        num_encoder_layers=6,
        num_heads=8,
        mlp_ratio=4.0
    )
    x = torch.randn(2, 1000, 80)
    x_lens = torch.tensor([1000, 900])
    target = torch.empty(2, 527).random_(2)

    loss = model(x, x_lens, target)
    print(loss)