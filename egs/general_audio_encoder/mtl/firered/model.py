import os

import torch

from feature_extractor import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed

def load_fireredasr_aed_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model

class FireRedEncoder(torch.nn.Module):
    def __init__(self, model_dir: str) -> None:
        super().__init__()
        cmvn_path = model_dir + "/cmvn.ark"
        self.feature_extractor = ASRFeatExtractor(cmvn_path)
        model_path = os.path.join(model_dir, "model.pth.tar")
        model = load_fireredasr_aed_model(model_path)
        self.model = model.encoder
        
        self.model.eval()
        
    def forward(self, wav_path: list, start_list: list, dur_list: list):
        device = next(self.parameters()).device
        feats, lengths, durs = self.feature_extractor(wav_path, start_list, dur_list)
        feats = feats.to(device)
        lengths = lengths.to(device)
        
        enc_outputs, enc_out_len, enc_mask = self.model(feats, lengths)
        return enc_outputs, enc_out_len, enc_mask
    
    def get_embeddings(self, wav_path_list, start_list, dur_list):
        return self.forward(wav_path_list, start_list, dur_list)
        