# modeling_spear.py

import torch

from transformers import PreTrainedModel, AutoConfig
from configuration_spear import SpearConfig
from spear_model import SpearModel as model


class SpearModel(PreTrainedModel):
    config_class = SpearConfig

    def __init__(self, config: SpearConfig):
        super().__init__(config)
        self.model = model(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_audio(self, audio_path):
        return self.model.load_audio(audio_path)
    
    @classmethod
    def from_legacy_checkpoint(cls, path, config):
        model = cls(config)
        ckpt = torch.load(path)["model"]
        info = model.model.model.load_state_dict(ckpt, strict=False)
        print(info)
        return model
    
def _test():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    audio_file = [
        # "common_voice_af_39597042.wav",
        "download/common_voice_17_0/audio/af/train/af_train_0/common_voice_af_39597046.wav",
    ]
    
    ckpt = "/mnt/shared-storage-user/housiyuan/xiaoyu/models/spear_encoders/327M-uni-v2-batch-mix-0.3-p-n0.5/iter-500000-avg-4.pt"
    config = SpearConfig()
    my_model = SpearModel.from_legacy_checkpoint(ckpt, config)
    my_model.eval()
    my_model.to(device)
    
    import pdb; pdb.set_trace()
    audio, audio_len = my_model.load_audio(audio_file)
    audio = audio.to(device)
    audio_len = audio_len.to(device)
    with torch.no_grad():
        outputs = my_model(audio, audio_len)
    
    encoder_out = outputs["encoder_out"] # (N,T,C)
    encoder_out_lens = outputs["encoder_out_lens"] # (N)
    middle_out = outputs["intermediate_hidden_states"] # list of (N,T,C)
    
    print(encoder_out)
    print(encoder_out_lens)
    print(middle_out[0].shape)
    
def _test_from_pretrained():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    audio_file = [
        # "common_voice_af_39597042.wav",
        "download/common_voice_17_0/audio/af/train/af_train_0/common_voice_af_39597046.wav",
    ]
    
    from transformers import AutoModel
    
    my_model = AutoModel.from_pretrained("/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl/spear_large_speech_hf", trust_remote_code=True)
    my_model.eval()
    my_model.to(device)
    
    audio, audio_len = my_model.load_audio(audio_file)
    audio = audio.to(device)
    audio_len = audio_len.to(device)
    with torch.no_grad():
        outputs = my_model(audio, audio_len)
    
    encoder_out = outputs["encoder_out"] # (N,T,C)
    encoder_out_lens = outputs["encoder_out_lens"] # (N)
    middle_out = outputs["intermediate_hidden_states"] # list of (N,T,C)
    
    print(encoder_out)
    print(encoder_out_lens)
    print(middle_out[0].shape)
    
    
if __name__=="__main__":
    _test_from_pretrained()
    
    
    