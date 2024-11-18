import math
from typing import Optional, List, Callable, Dict, Any

from lhotse import CutSet, validate
from lhotse.dataset.collation import collate_audio
from lhotse.audio.utils import suppress_audio_loading_errors
import torch

class WaveformAsrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        collate: bool = True,
    ):
        super().__init__()
        self.collate = collate
        
    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        texts = [c.supervisions[0].text for c in cuts]
        if self.collate:
            audio, audio_lens = collate_audio(cuts)
            return {
                "cuts": cuts,
                "audio": audio,
                "audio_lens": audio_lens,
                "text": texts,
            }
        else:
            remain_cuts = []
            remain_audios = []
            audio_lens = []
            for c in cuts:
                with suppress_audio_loading_errors():
                    cur_audio = c.load_audio()
                    remain_audios.append(cur_audio)
                    audio_lens.append(len(cur_audio))
                    remain_cuts.append(c)
            return {"cuts": CutSet.from_cuts(remain_cuts), "audio": remain_audios, "audio_lens": audio_lens, "text": texts}
    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)