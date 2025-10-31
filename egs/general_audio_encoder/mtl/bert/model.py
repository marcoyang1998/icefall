from typing import List

from transformers import BertTokenizer, BertModel

import torch

CKPT_DICT = {
    "large": "/mnt/shared-storage-user/brainllm-share/checkpoints/bert-large-uncased",
    "base": "/mnt/shared-storage-user/brainllm-share/checkpoints/bert-base-uncased",
}

class MyBertModel(torch.nn.Module):
    def __init__(self, model_version: str = "large") -> None:
        super().__init__()
        if model_version == "large":
            model_dir = CKPT_DICT["large"]
            model_dim = 1024
        elif model_version == "base":
            model_dir = CKPT_DICT["base"]
            model_dim = 768
        else:
            raise ValueError(f"Unsupported model_version: {model_version}")
        
        self.model_version = model_version
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)
        self.model_dim = model_dim
    
    def get_embeddings(
        self,
        texts: List[str],
        layer_idx: int = -1,
    ):
        assert layer_idx == -1, "Currently only support getting the last layer's embedding"
        
        device = next(self.model.parameters()).device
        
        encoded_input = self.tokenizer(texts, padding=True, return_tensors='pt', return_length=True)
        lengths = encoded_input.pop("length")
        
        encoded_input.to(device)
        outputs = self.model(**encoded_input)
        last_hidden_state = outputs.last_hidden_state  # (B, L, C)
        return last_hidden_state, lengths
    
    
    
if __name__ == "__main__":
    model_version = "base"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = MyBertModel(model_version=model_version)
    model.eval()
    model.to(device)
    
    texts = ["Hello, this is Xiaoyu", "my name is Xiaoyu"]
    embeddings, embedding_lens = model.get_embeddings(texts)
    print(embeddings.shape)
    print(embedding_lens)