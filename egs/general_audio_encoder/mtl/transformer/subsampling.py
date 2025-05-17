import torch
import torch.nn as nn

class WhisperConv1d(nn.Module):
    def __init__(self, n_mels=128, n_state=768):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        

class Conv2dSubsampling(nn.Module):
    def __init__(self, 
        idim: int = 128,
        intermediate_dim: int = 128,
        odim: int = 768,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=intermediate_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=intermediate_dim, out_channels=intermediate_dim, kernel_size=3, stride=2),
            nn.GELU(),
        )
        self.out = nn.Linear(intermediate_dim * ((idim - 1) // 2), odim)
        
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).
          x_lens:
            Length of the input feature (N,)

        Returns:
          Return a tensor of shape (N, ((T-1)//2 , odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, T, odim)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x_lens = (x_lens - 1) // 2
        # Now x is of shape (N, (T-1)//2 , odim)
        return x, x_lens
    
if __name__=="__main__":
    conv_subsample = Conv2dSubsampling(idim=128, odim=768)
    num_params = sum([p.numel() for p in conv_subsample.parameters()])
    print(f"Number of model parameters: {num_params}")
    num_params = sum([p.numel() for p in conv_subsample.out.parameters()])
    print(f"Number of model parameters: {num_params}")
    whisper_conv = WhisperConv1d()
    num_params = sum([p.numel() for p in whisper_conv.parameters()])
    print(f"Number of whisper conv parameters: {num_params}")
    
    x = torch.rand(2,200, 128)
    x_lens = torch.tensor([200, 190])
    y, y_lens = conv_subsample(x, x_lens)
    print(y.shape)
    print(y_lens)
    