import torch
import logging
from icefall.utils import make_pad_mask

class Teacher(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module
    ):
        super().__init__()
        self.model = model
        device = next(model.parameters()).device
        logging.info(f"The teacher model is on device: {device}")
        
        
    def get_embeddings(self):
        raise NotImplementedError()
    
class BEATsTeacher(Teacher):
    def __init__(
        self,
        model: torch.nn.Module,
    ):
        super().__init__(model)
        
    @torch.no_grad()
    def get_embeddings(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
    ):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        padding_mask = make_pad_mask(audio_lens)
        representation = self.model.extract_features(
            audio.to(device).to(dtype),
            padding_mask=padding_mask.to(device),
        )[0]
        return representation
        
        
class EcapaTeacher(Teacher):
    def __init__(
        self,
        model: torch.nn.Module,
    ):
        super().__init__(model)
    
    @torch.no_grad()
    def get_embeddings(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
    ):
        representation = self.model.encode_batch(
            wavs=audio,
            wav_lens=audio_lens/torch.max(audio_lens)
        )
        return representation
    
if __name__=="__main__":
    import torchaudio
    from BEATs import BEATs, BEATsConfig
    
    beats_ckpt = "data/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    checkpoint = torch.load(beats_ckpt)
    cfg = BEATsConfig(checkpoint['cfg'])
    
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    
    audio_input_16khz, _ = torchaudio.load('/star-fj/fangjun/open-source/icefall/egs/librispeech/ASR/download/LibriSpeech/train-clean-100/6078/54007/6078-54007-0023.flac')
    padding_mask = torch.zeros_like(audio_input_16khz).bool()
    
    features = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
    print(features.shape)
    