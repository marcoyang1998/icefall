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
        )[0]
        return representation