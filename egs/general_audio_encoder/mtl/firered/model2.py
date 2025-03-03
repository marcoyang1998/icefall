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
        import pdb; pdb.set_trace()
        feats, lengths, durs = self.feature_extractor(wav_path, start_list, dur_list)
        feats = feats.to(device)
        lengths = lengths.to(device)
        
        enc_outputs, enc_out_len, enc_mask = self.model(feats, lengths)
        return enc_outputs, enc_out_len, enc_mask
    
    def get_embeddings(self, wav_path_list, start_list, dur_list):
        return self.forward(wav_path_list, start_list, dur_list)
        
        
if __name__=="__main__":
    model_dir = "/fs-computility/INTERN6/shared/yangxiaoyu/workspace/FireRedASR/pretrained_models/FireRedASR-AED-L"
    model = FireRedEncoder(model_dir=model_dir)
    model.eval()
    device =torch.device("cuda")
    model.to(device)
    
    from lhotse import load_manifest_lazy
    cuts = load_manifest_lazy("data/fbank_wenetspeech_wav_trimmed/wenetspeech_cuts_S.jsonl.gz")
    
    for cut in cuts:
        import pdb; pdb.set_trace()
        recording = cut.recording.sources[0].source
        wav_path = [recording]
        start = [cut.start]
        duration = [cut.duration]
        
        enc_out, _, _ = model(wav_path, start, duration)
    
    # wav_trimmed = "test_trimmed.wav"
    
    
    # enc_out2, _, _ = model([wav_trimmed], [0.0], [2.0])
    
    # print(enc_out)
    # print(enc_out2)