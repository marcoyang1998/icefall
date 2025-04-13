from typing import Dict

from transformers import (
    AutoProcessor, 
    Data2VecAudioModel,
    AutoModel,
    Wav2Vec2FeatureExtractor,
    AutoFeatureExtractor,
    Wav2Vec2BertModel,
)
import torch

class HubertModel(torch.nn.Module):
    def __init__(self, model_version: str="large", normalize: bool=True):
        super().__init__()
        if model_version == "large":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/hubert-{model_version}-ll60k")
            self.model = AutoModel.from_pretrained(f"facebook/hubert-{model_version}-ll60k")
        elif model_version == "base":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/hubert-{model_version}-ls960")
            self.model = AutoModel.from_pretrained(f"facebook/hubert-{model_version}-ls960")
        # self.processor.do_normalize = normalize
    
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
        layer_results = all_layer_results[layer_idx] # (N,T,C)
        padding_mask = self.model._get_feature_vector_attention_mask(layer_results.shape[1], inputs["attention_mask"])
        embedding_lens = padding_mask.sum(dim=1)
        
        return layer_results, embedding_lens
    
    def get_embeddings(
        self,
        batch: Dict,
        layer_idx: int,
    ):
        return self.extract_features(batch, layer_idx)
    
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