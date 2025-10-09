import argparse
import torch

class ZipformerConfig:
    def __init__(self):
        # 用 _config 存储所有参数
        self._config = {
            "feature_dim": 128,
            "pos_dim": 48,
            "output_downsampling_factor": 2,
            "downsampling_factor": "1,2,4,8,4,2",
            "num_encoder_layers": "2,2,3,4,3,2",
            "feedforward_dim": "512,768,1024,1536,1024,768",
            "encoder_dim": "192,256,448,768,448,192",
            "encoder_unmasked_dim": "192,192,256,256,256,192",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "num_heads": "4,4,4,8,4,4",
            "causal": True,
        }

    def __getattr__(self, key):
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"'ZipformerConfig' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "_config":
            super().__setattr__(key, value)
        else:
            self._config[key] = value

    def __delattr__(self, key):
        if key in self._config:
            del self._config[key]
        else:
            raise AttributeError(f"'ZipformerConfig' object has no attribute '{key}'")

    def to_dict(self):
        return dict(self._config)

    def __repr__(self):
        return f"ZipformerConfig({self._config})"
        


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
    
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)