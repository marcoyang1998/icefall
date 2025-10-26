from transformers import PretrainedConfig


class SpearConfig(PretrainedConfig):
    model_type = "spear"

    def __init__(
        self,
        num_mel_bins: int = 128,
        pos_dim: int = 48,
        output_downsampling_factor: int = 1,
        downsampling_factor: str = "1,2,4,8,4,2,1",
        num_encoder_layers: str = "1,2,2,3,1,1,1",
        feedforward_dim: str = "3072,3072,3072,3072,3072,3072,3072",
        encoder_dim: str = "1024,1024,1024,1024,1024,1024,1024",
        encoder_unmasked_dim: str = "512,512,512,512,512,512,512",
        cnn_module_kernel: str = "31,31,15,15,15,31,31",
        num_heads: str = "8,8,8,8,8,8,8",
        causal: bool = False,
        chunk_size: int = 8,
        left_context_frames: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_downsampling_factor = output_downsampling_factor
        self.num_mel_bins = num_mel_bins
        self.pos_dim = pos_dim
        self.downsampling_factor = downsampling_factor
        self.num_encoder_layers = num_encoder_layers
        self.feedforward_dim = feedforward_dim
        self.encoder_dim = encoder_dim
        self.encoder_unmasked_dim = encoder_unmasked_dim
        self.cnn_module_kernel = cnn_module_kernel
        self.num_heads = num_heads
        
        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames
        