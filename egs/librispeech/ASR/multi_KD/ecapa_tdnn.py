import math

import torch
from torch import nn
import torch.nn.functional as F

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling as AttentiveStatisticsPooling_orig

def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

class _Conv1d(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        # This is just a wrapper around the nn.Conv1d
        # For model loading compatibility
        super(_Conv1d,self).__init__()
        self.conv = nn.Conv1d(**kwargs)
    def forward(self, x):
        return self.conv(x)
    
class _BatchNorm1d(nn.Module):
    def __init__(self, **kwargs):
        super(_BatchNorm1d, self).__init__()
        self.norm = nn.BatchNorm1d(**kwargs)
        
    def forward(self, x):
        return self.norm(x)

class TDNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = _Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            padding="same",
            padding_mode="reflect"
        )
        self.activation = activation()
        self.norm = _BatchNorm1d(num_features=out_channels)
    
    def forward(self, x):
        # input shape (N,C,L)
        x = self.activation(self.conv(x))
        x = self.norm(x)
        return x

class AttentiveStatisticsPooling(nn.Module):
    """Copy from https://github.com/speechbrain/speechbrain/blob/46e282f46e3047fca30fc914d0cb39ab2684238d/speechbrain/lobes/models/ECAPA_TDNN.py#L198
    Make small changes accordingly

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """
    def __init__(self, channels, attention_channels=128, global_context=True):
        super(AttentiveStatisticsPooling, self).__init__()
        
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = _Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1, 
            padding="same",
            padding_mode="reflect",
        )
        
    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1).to(x.dtype)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).to(x.dtype)
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x
        

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats
        
        
if __name__=="__main__":
    x = torch.rand(1,120,64).permute(0,2,1)
    lengths = torch.rand((1,))
    asp = AttentiveStatisticsPooling(64)
    asp_orig = AttentiveStatisticsPooling_orig(64)
    
    # load the original state dict
    state_dict = asp_orig.state_dict()
    asp.load_state_dict(state_dict)
    
    y_orig = asp_orig(x, lengths).transpose(1,2)
    y = asp(x, lengths).transpose(1,2)
    import pdb; pdb.set_trace()
    print(y)