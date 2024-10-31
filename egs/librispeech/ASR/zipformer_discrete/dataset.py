import math
from typing import Optional, List, Callable, Dict, Any

from lhotse import CutSet, validate
from lhotse.utils import compute_num_frames, ifnone
import torch
from torch.nn.utils.rnn import pad_sequence

class DiscretizedInputSpeechRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech recognition task that provides discrete audio tokens instead of audios/features.
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'token': (B x Tokens) int tensor
            'token_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        field: str,
        num_tokens: int,
        token_type: str,
        frequency_size: Optional[int] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.field = field
        self.num_tokens = num_tokens
        self.frequency_size = frequency_size
        self.token_type = token_type
        self.input_transforms = ifnone(input_transforms, [])

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        if self.token_type in ("wavlm", "hubert"):
            tokens = []
            token_lens = []
            for c in cuts:
                token = torch.tensor(
                    c.tokens, dtype=torch.int64
                )
                tokens.append(token)
                token_lens.append(token.size(0))
            tokens = pad_sequence(
                tokens, batch_first=True, padding_value=self.num_tokens
            )
            token_lens = torch.tensor(token_lens, dtype=torch.int64)
        elif self.token_type == "vq-wav2vec":
            tokens = []
            token_lens = []
            for c in cuts:
                token = torch.tensor(
                    list(map(int, c.discrete_tokens.split())), dtype=torch.int64
                )
                token_len = len(token) >> 1
                tokens.append(token.reshape(2, token_len).T)
                token_lens.append(token_len)
            tokens = pad_sequence(
                tokens, batch_first=True, padding_value=self.num_tokens
            )
            token_lens = torch.tensor(token_lens, dtype=torch.int64)
        elif self.token_type == "encodec":
            tokens = []
            token_lens = []
            for c in cuts:
                token = torch.tensor(
                    list(map(int, c.discrete_tokens.split())), dtype=torch.int64
                )
                token_len = len(token) >> 3
                tokens.append(token.reshape(8, token_len).T)
                token_lens.append(token_len)
            tokens = pad_sequence(
                tokens, batch_first=True, padding_value=self.num_tokens
            )
            token_lens = torch.tensor(token_lens, dtype=torch.int64)

        if self.token_type in ("wavlm", "hubert"):
            tokens = (
                torch.nn.functional.interpolate(
                    tokens.unsqueeze(0).to(torch.float32),
                    size=int(tokens.size(1)) * 2,
                    mode="nearest",
                )
                .squeeze(0)
                .to(torch.int64)
            )
            token_lens = token_lens * 2
        elif self.token_type == "encodec":
            tokens = (
                torch.nn.functional.interpolate(
                    tokens.unsqueeze(0).to(torch.float32),
                    size=(math.ceil(tokens.size(1) * 4 / 3), 8),
                    mode="nearest",
                )
                .squeeze(0)
                .to(torch.int64)
            )
            token_lens = (tokens[:, :, 0] != self.num_tokens).sum(1)

        data_dict = {}
        for tnfm in self.input_transforms:
            if tnfm.__class__.__name__ == "DiscretizedInputAugment":
                tokens, frequency_masks = tnfm(
                    tokens, self.num_tokens, self.frequency_size
                )
                data_dict["frequency_masks"] = frequency_masks
            else:
                tokens = tnfm(tokens)

        data_dict["cuts"] = cuts
        data_dict["tokens"] = tokens
        data_dict["token_lens"] = token_lens

        return data_dict

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)