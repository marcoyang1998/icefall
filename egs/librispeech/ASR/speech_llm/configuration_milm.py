""" MiLM model configuration"""
from transformers import PretrainedConfig


class MiConfig(PretrainedConfig):
    model_type = "milm"

    attribute_map = {
        "vocab_size": "num_embeddings",
        "hidden_size": "decoder_embed_dim",
        "n_layer": "decoder_layers",
        "n_head": "decoder_attention_heads",
    }

    def __init__(
        self,
        num_embeddings=60064,
        decoder_embed_dim=2048,
        decoder_layers=1,
        decoder_attention_heads=32,
        qk_bias=True,
        use_fused=False,
        use_flash=False,
        n_positions=4096,
        padding_idx=0,
        bos_token_id=1,
        eos_token_id=2,
        scale_token_embedding=False,
        decoder_kv_heads=None,
        qkv_pack=False,
        tie_weights=True,
        multiple_of=256,
        ffn_dim_multiplier=2.5,
        rope_theta=10000,
        torch_dtype="float16",
        device=None,
        rope_scaling=None,
        tp_size=1,
        quant=None,
        n_bits=None,
        group_size=None,
        quantization_bit=None,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.num_embeddings = num_embeddings
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.qk_bias = qk_bias
        self.use_fused = use_fused
        self.use_flash = use_flash
        self.n_positions = n_positions
        self.padding_idx = padding_idx
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_kv_heads = decoder_kv_heads
        self.rope_theta = rope_theta
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.shard_embedding = tie_weights
        self.scale_token_embedding = scale_token_embedding
        self.qkv_pack = qkv_pack
        self.multiple_of = multiple_of
        self.torch_dtype = torch_dtype
        self.device = device
        self.rope_scaling = rope_scaling
        self.tp_size = tp_size
        self.quant = quant
        self.n_bits = n_bits
        self.group_size = group_size
        self.quantization_bit = quantization_bit
