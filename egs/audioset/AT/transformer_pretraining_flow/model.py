from functools import partial
from json import encoder

import torch
import torch.nn as nn

#from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible, get_1d_sincos_pos_embed_from_grid
from util.misc import concat_all_gather
from util.patch_embed import PatchEmbed_new, PatchEmbed_org

class AudioMAEModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        embed_dim: int,
        decoder_dim: int,
        input_dim: int=80,
        patch_width: int=4,
    ):
        self.input_dim = input_dim
        self.patch_width = patch_width
        
        # encoder related
        if use_custom_patch:
            print(f'Use custom patch_emb with patch size: {patch_size}, stride: {stride}')
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        else:
            self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)

        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.encoder_out_proj = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        # decoder related
        self.decoder = decoder
        self.decoder_dim = decoder_dim
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim), requires_grad=pos_trainable)  
        self.decoder_pred = nn.Linear(decoder_dim, patch_width * input_dim, bias=True)

        # we don't use cls token

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
        return: (B, T//4, C)
        """
        pass
    
    def forward_encoder(self, x):
        x = self.patch_em
