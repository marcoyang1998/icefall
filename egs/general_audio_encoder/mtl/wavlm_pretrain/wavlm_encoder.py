import torch
from WavLM import WavLM, WavLMConfig

from icefall.utils import make_pad_mask

CKPT_DICT = {
    "base": "download/models/WavLM-Base.pt",
    "base+": "download/models/WavLM-Base+.pt",
    "large": "download/models/WavLM-Large.pt",
}

class WavlmModel(torch.nn.Module):
    def __init__(self, model_version: str="large"):
        super().__init__()
        assert model_version in ["base", "base+", "large"]
        # we only borrow the config of the model, throw away the state dict
        ckpt_path = CKPT_DICT[model_version]
        checkpoint = torch.load(ckpt_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        
        self.cfg = cfg
        self.model = model
        
    def extract_features(
        self,
        batch,
        layer_idx: int,
    ):
        device = next(self.model.parameters()).device
        
        audio_input_16khz = batch["audio"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        padding_mask = make_pad_mask(audio_lens)
        
        if self.cfg.normalize:
            audio_input_16khz = torch.nn.functional.layer_norm(audio_input_16khz, audio_input_16khz.shape)
        
        (rep, layer_results), padding_mask = self.model.extract_features(
            audio_input_16khz,
            padding_mask=padding_mask,
            output_layer=self.model.cfg.encoder_layers,
            ret_layer_results=True
        )
        
        layer_results = [res.permute(1,0,2) for res, _ in layer_results] # list of (B,T,C)
        layer_results = layer_results[layer_idx] # (B,T,C)
        embedding_lens = (~padding_mask).sum(dim=-1)
        
        return layer_results, embedding_lens
    
    def get_embeddings(
        self, 
        batch: dict,
        layer_idx: int,
    ):
        return self.extract_features(batch, layer_idx)
    
    def forward(self, x, x_lens, padding_mask):
        # x: input audio
        # x_lens: length of the inpu taudio
        
        if self.cfg.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)
        
        padding_mask = make_pad_mask(x_lens)
        
        x, padding_mask = self.model.extract_features(
            x,
            padding_mask=padding_mask,
            output_layer=self.model.cfg.encoder_layers,
        )
        
        x_lens = (~padding_mask).sum(dim=-1)
        
        return x, x_lens
