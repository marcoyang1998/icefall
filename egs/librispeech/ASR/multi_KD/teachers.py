import logging
from typing import List

import torch
import torch.nn.functional as F
import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
import numpy as np
import torchaudio.transforms as T

from icefall.utils import make_pad_mask


class Teacher(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module
    ):
        super().__init__()
        self.model = model
        
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
    
class WhisperTeacher(Teacher):
    def __init__(
        self,
        model: torch.nn.Module,
        n_mels: int = 80,
    ):
        super().__init__(model)
        self.n_mels = n_mels
        
    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        layer_idx: int = -1,
    ):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        x_lens: torch.Tensor, shape = (batch_size)
        layer_idx: which layer's feature to extract
        """
        x = F.gelu(self.model.conv1(x))
        x = F.gelu(self.model.conv2(x))
        x = x.permute(0, 2, 1)
        x_lens = torch.floor((x_lens + 1)/2).int()
        
        # make the model compatible with any input length
        mask = make_pad_mask(x_lens, max_len=1500).to(x.device)
        pos_emb = self.model.positional_embedding.masked_fill(mask.unsqueeze(-1), 0.0)
        x = (x + pos_emb[:,:x_lens.max(),:]).to(x.dtype)
        
        results = []
        for block in self.model.blocks:
            x = block(x)
            results.append(x)
        if layer_idx == -1: # use the last layer
            x = self.model.ln_post(x)
        else:
            x = results[layer_idx] # zero-based index

        return x, x_lens
            
    @torch.no_grad()
    def get_embeddings(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        layer_idx: int = -1
    ):
        # return the embeddings of the input audio
        # audio_lens is the number of raw samples (i.e waveform)
        device = next(self.model.parameters()).device
        audio = audio.to(device)
        audio_lens = audio_lens.to(device)
        mel = log_mel_spectrogram(audio, n_mels=self.n_mels) # (N, n_mel, T)
        
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        
        mel_lens = torch.floor(audio_lens/160).int()
        assert mel_lens.max() <= mel.size(-1)

        features, feature_lens = self.forward_encoder(
            mel,
            mel_lens,
            layer_idx=layer_idx,
        )
        
        return features, feature_lens
        
class MertTeacher(Teacher):
    def __init__(
        self,
        model: torch.nn.Module,
        processor: torch.nn.Module,
    ):
        super().__init__(model)
        self.processor = processor
        self.resample_rate = processor.sampling_rate

    def forward_processor(
        self, audio: np.array, sampling_rate: int,
    ):
        # Here we assume the audio is in the correct sampling rate
        inputs = self.processor(audio, padding=True, sampling_rate=sampling_rate, return_tensors="pt")
        return inputs
    
    @torch.no_grad()
    def get_embeddings(
        self,
        audio: np.array,
        sampling_rate: int,
        level: str="frame",
        layer_idx: int=-1,
    ):
        device = next(self.model.parameters()).device
        inputs = self.forward_processor(audio, sampling_rate=sampling_rate)
        inputs = inputs.to(device)
        outputs = self.model(**inputs, output_hidden_states=True)

        # (num_layers, B, T, feature_dim)
        all_layer_hidden_states = torch.stack(outputs.hidden_states)
        
        if level == "clip":
            features = all_layer_hidden_states.mean(-2) # (num_layers, B, feature_dim)
        elif level == "frame":
            features = all_layer_hidden_states
        else:
            raise NotImplementedError()

        return features[layer_idx, ...]


    
if __name__=="__main__":
    import torchaudio
    from BEATs import BEATs, BEATsConfig
    
    # beats_ckpt = "data/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    # checkpoint = torch.load(beats_ckpt)
    # cfg = BEATsConfig(checkpoint['cfg'])
    
    # BEATs_model = BEATs(cfg)
    # BEATs_model.load_state_dict(checkpoint['model'])
    # BEATs_model.eval()
    
    # audio_input_16khz, _ = torchaudio.load('/star-fj/fangjun/open-source/icefall/egs/librispeech/ASR/download/LibriSpeech/train-clean-100/6078/54007/6078-54007-0023.flac')
    # padding_mask = torch.zeros_like(audio_input_16khz).bool()
    
    # features = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
    # print(features.shape)
    
    
    audio = "/star-xy/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
    audio = torchaudio.load(audio)[0]
    audio = torch.cat([audio, audio],dim=0)
    
    audio_lens = torch.tensor([93680, 92800])
    
    model = whisper.load_model("base")
    device = model.device
    teacher = WhisperTeacher(model.encoder)
    
    teacher.get_embeddings(
        audio.to(device),
        audio_lens.to(device),
        layer_idx=6,
    )
    