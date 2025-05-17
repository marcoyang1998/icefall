import logging
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding, eager_attention_forward
)
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPast

from icefall.utils import make_pad_mask

class LlamaAudioEncoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 768,
        num_layers: int = 10,
        num_attention_heads: int = 8,
        hidden_act: str = "gelu",
        use_flash_attention: bool = True,
        is_causal: bool = False,
    ):
        # a Llama Audio Encoder model with rotary positional embedding (ROPE)
        # supports both streaming and non-streaming by specifying is_causal
        super().__init__()
        
        if use_flash_attention:
            attn_implementation="flash_attention_2"
        else:
            attn_implementation="eager"
        
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        
        config = LlamaConfig(
            hidden_size=encoder_dim,
            intermediate_size=encoder_dim * 4,       # 通常是 2-4 倍 hidden_size（可自调）
            num_hidden_layers=num_layers,
            vocab_size=10,
            num_attention_heads=num_attention_heads,       # 必须整除 hidden_size
            max_position_embeddings=2048, # 有RoPE时这个可大些
            hidden_act=hidden_act,            # LLaMA 默认是 SiLU
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            attn_implementation=attn_implementation,
        )
        self.config = config
        self.is_causal = is_causal
        
        self.layers = nn.ModuleList(
            [LlamaEncoderLayer(config, layer_idx, is_causal) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self, 
        inputs_embeds: torch.Tensor,
        input_lens: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        # Performs forward of audio features
        cache_position = torch.arange(
            0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the encoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        attention_mask = ~make_pad_mask(input_lens) # the llama attention mask is flipped
        
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
        
        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
        return output

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # Adapted from transformers, added the option of being non-causal

    def __init__(self, config: LlamaConfig, layer_idx: int, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = is_causal # controls the attention mechanism

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logging.warning(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaEncoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, is_causal: bool = False):
        
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_causal = is_causal

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx, is_causal=is_causal)
        
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
def _test_padding_mask():
    device = torch.device("cuda")
    
    model = LlamaAudioEncoder(
        encoder_dim=512,
        num_layers=12,
        use_flash_attention=True,
        is_causal=False,
    )
    model.eval()
    model.to(device)
    
    x = torch.rand(1,8,512).to(device)
    x = x.repeat(3,1,1) 
    x[1, 6:, : ] = torch.rand(2,512)
    x[2, 6:, : ] = torch.rand(2,512)
    x_lens = torch.tensor([8, 5, 5]).to(device)
    
    with torch.cuda.amp.autocast(enabled=True):
        output = model(
            inputs_embeds=x,
            input_lens=x_lens,
        )
    features = output.last_hidden_state
    assert torch.all(features[1, :5] == features[2, :5])
    print("Check for padding mask: passed!")
    
    
def _test():
    device = torch.device("cuda")
    
    model = LlamaAudioEncoder(
        encoder_dim=512,
        num_layers=12,
        use_flash_attention=True,
        is_causal=False,
    )
    model.eval()
    model.to(device)
    
    x = torch.rand(1,8,512).to(device)
    x = x.repeat(3,1,1) 
    x[1, 1:2, : ] = torch.rand(1,512)
    x_lens = torch.tensor([8, 8, 5]).to(device)
    
    with torch.cuda.amp.autocast(enabled=True):
        output = model(
            inputs_embeds=x,
            input_lens=x_lens,
        )
    features = output.last_hidden_state
    # you should expect that first and second to be close
    # but first and last to be more different
    print(features[0])
    print(features[1])
    print(features[2])
    
def _test_causal():
    device = torch.device("cuda")
    is_causal = True
    
    model = LlamaAudioEncoder(
        encoder_dim=512,
        num_layers=12,
        use_flash_attention=True,
        is_causal=is_causal,
    )
    model.eval()
    model.to(device)
    
    x = torch.rand(1,8,512).to(device)
    x = x.repeat(3,1,1) 
    x[1, 2:3, : ] = torch.rand(1,512)
    x_lens = torch.tensor([8, 8, 5]).to(device)
    
    with torch.cuda.amp.autocast(enabled=True):
        output = model(
            inputs_embeds=x,
            input_lens=x_lens,
        )
    features = output.last_hidden_state
    assert torch.all(features[0,:5] == features[2, :5])
    assert torch.all(features[0,:2] == features[1, :2])
    print("Checking causal mask: passed!")
    
    

if __name__=="__main__":
    _test_padding_mask()
    _test_causal()
    _test()
    