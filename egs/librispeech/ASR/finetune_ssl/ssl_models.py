from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
from transformers import (
    AutoProcessor, 
    Data2VecAudioModel,
    AutoModel,
    Wav2Vec2FeatureExtractor,
    AutoFeatureExtractor,
    Wav2Vec2BertModel,
)

from icefall.utils import make_pad_mask
from WavLM import WavLM, WavLMConfig

class Teacher(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module
    ):
        super().__init__()
        self.model = model
        
    def get_embeddings(self):
        raise NotImplementedError()

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
        if layer_idx == -1 or layer_idx == len(results) - 1: # use the last layer
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
    
class HuggingfaceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
    
    def prepare_input_data(
        self,
        batch,
    ):
        audio_pt = batch["audio"]
    
        if isinstance(audio_pt, list):
            audios = [audio.numpy() for audio in audio_pt]
        else:
            audio_lens_pt = batch["audio_lens"]
            audios = []
            for i in range(audio_pt.shape[0]):
                audios.append(audio_pt[i, :audio_lens_pt[i]].tolist())
        return audios
    
    def extract_features(
        self,
        batch: Dict,
        layer_idx: int,
    ):
        audios = self.prepare_input_data(batch)
        # the audios should be a list of np array, without padding
        device = next(self.model.parameters()).device
        
        inputs = self.processor(
            audios, 
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)
        
        outputs = self.model(
            output_hidden_states=True,
            **inputs,
        )
        all_layer_results = outputs.hidden_states
        layer_results = all_layer_results[layer_idx].cpu().numpy() # (N,T,C)
        padding_mask = self.model._get_feature_vector_attention_mask(layer_results.shape[1], inputs["attention_mask"])
        embedding_lens = padding_mask.sum(dim=1)
        
        return layer_results, embedding_lens
    
    def forward(self, batch):
        audios = self.prepare_input_data(batch)
        # the audios should be a list of np array, without padding
        device = next(self.model.parameters()).device
        inputs = self.processor(
            audios, 
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)
        
        outputs = self.model(
            output_hidden_states=True,
            **inputs,
        )
        features = outputs.last_hidden_state
        all_layer_results = outputs.hidden_states
        
        padding_mask = self.model._get_feature_vector_attention_mask(features.shape[1], inputs["attention_mask"])
        embedding_lens = padding_mask.sum(dim=1)
        
        return features, all_layer_results, embedding_lens
        

class Data2Vec(HuggingfaceModel):
    def __init__(self, model_version: str="large"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(f"facebook/data2vec-audio-{model_version}")
        self.model = Data2VecAudioModel.from_pretrained(f"facebook/data2vec-audio-{model_version}")
    
    
class HuBERT(HuggingfaceModel):
    def __init__(self, model_version: str="large", normalize: bool=True):
        super().__init__()
        if model_version == "large":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/hubert-{model_version}-ll60k")
            self.model = AutoModel.from_pretrained(f"facebook/hubert-{model_version}-ll60k")
        elif model_version == "base":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/hubert-{model_version}-ls960")
            self.model = AutoModel.from_pretrained(f"facebook/hubert-{model_version}-ls960")
        self.processor.do_normalize = normalize
        
class W2vBERT(HuggingfaceModel):
    def __init__(self, model_version: str="large"):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        
class WavlmModel(torch.nn.Module):
    def __init__(self, ckpt_path: str="models/WavLM-Large.pt"):
        super().__init__()
        checkpoint = torch.load(ckpt_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        
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
        
        layer_results = [res.permute(1,0,2).cpu().numpy() for res, _ in layer_results] # list of (B,T,C)
        layer_results = layer_results[layer_idx] # (B,T,C)
        embedding_lens = (~padding_mask).sum(dim=-1)
        
        return layer_results, embedding_lens
    
    def forward(self, batch):
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
        layer_results = [res.permute(1,0,2) for res, _ in layer_results]
        embedding_lens = (~padding_mask).sum(dim=-1)
        
        return rep, layer_results, embedding_lens
        