import math
from typing import Optional, List, Callable, Dict, Any

from lhotse import CutSet, validate
from lhotse.dataset.collation import collate_audio
from lhotse.audio.utils import suppress_audio_loading_errors
from lhotse.utils import compute_num_frames, ifnone
import torch
from torch.nn.utils.rnn import pad_sequence
from lhotse.workarounds import Hdf5MemoryIssueFix

class WaveformAudioTaggingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        collate: bool = True,
    ):
        super().__init__()
        self.collate = collate
        
    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        audio_events = [c.supervisions[0].audio_event for c in cuts]
        if self.collate:
            audio, audio_lens = collate_audio(cuts)
            return {
                "cuts": cuts,
                "audio": audio,
                "audio_lens": audio_lens,
                "audio_events": audio_events,
            }
        else:
            remain_cuts = []
            remain_audios = []
            for c in cuts:
                with suppress_audio_loading_errors():
                    remain_audios.append(c.load_audio())
                    remain_cuts.append(c)
            return {"cuts": CutSet.from_cuts(remain_cuts), "audio": remain_audios, "audio_events": audio_events}
    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

class DiscretizedInputAudioTaggingDataset(torch.utils.data.Dataset):
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
        duplicate_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.field = field
        self.num_tokens = num_tokens
        self.frequency_size = frequency_size
        self.token_type = token_type
        self.input_transforms = ifnone(input_transforms, [])
        self.duplicate_tokens = duplicate_tokens
        
        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

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

        if self.duplicate_tokens:
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
                
        audio_events = [cut.supervisions[0].audio_event for cut in cuts]

        data_dict["cuts"] = cuts
        data_dict["tokens"] = tokens
        data_dict["token_lens"] = token_lens
        data_dict["audio_events"] = audio_events

        return data_dict

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)