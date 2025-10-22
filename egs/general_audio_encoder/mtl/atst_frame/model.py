from typing import Dict
import torch

from audiossl.methods.atstframe.embedding import load_model, get_timestamp_embedding

CKPT="download/models/atstframe_base.ckpt"

class ATST_FrameEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = load_model(CKPT)
        model.eval()
        self.model = model
        self.embed_dim = 768

    def get_embeddings(
        self,
        audio,
        audio_lens,
        layer_idx: int = -1,
        concat_all_layers: bool = False,
    ):
        """Return the embeddings of ATST Encoder

        Args:
            audio (torch.tensor): input audio waveform
            audio_lens (torch.tensor): length of the input waveform
            layer_idx (int, optional): Which layer to return. If -1, we return the last layer.
            concat_all_layers (bool, optional): If we return the concatenation of all layers. Defaults to False.

        Returns:
            _type_: _description_
        """
        if concat_all_layers:
            n_blocks = 12
        else:
            n_blocks = 1
        embed,t = get_timestamp_embedding(audio, self.model, n_blocks=n_blocks) # (N, T, C*num_layers)
        embed = embed[:,:,-self.embed_dim * n_blocks:] # (N, T, C)
        embed_lens = (audio_lens / 16000 * 25).int() # the frame rate is 25 Hz
        return embed, embed_lens
        
    def extract_features(self, batch: Dict, layer_idx: int = -1):
        assert layer_idx == -1, "Currently only support last layer"
        device = next(self.model.parameters()).device
        
        audio = batch["audio"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        
        return self.get_embeddings(audio, audio_lens, layer_idx)


if __name__ == "__main__":
    model = ATST_FrameEncoder()
    x = torch.randn(2, 16000*10)
    x_lens = torch.tensor([16000*10, 16000*5])
    import pdb; pdb.set_trace()
    y, y_lens = model.get_embeddings(x, x_lens)
    import pdb; pdb.set_trace()
    print(y.shape) # (B,T,C)
    print(x_lens) # the frame rate is 25 Hz
    print(y.shape[1], x_lens) # y.shape[1] should be equal to x_lens