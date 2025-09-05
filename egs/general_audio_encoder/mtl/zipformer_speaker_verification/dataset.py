from typing import Callable, Dict, List, Union
import random

import numpy as np

from lhotse import CutSet, load_manifest, load_manifest_lazy
from lhotse import Fbank, FbankConfig
from lhotse.dataset import CutMix
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures, OnTheFlyFeatures
from lhotse.dataset.collation import read_audio_from_cuts, collate_matrices
from lhotse.cut import MonoCut
from lhotse.utils import LOG_EPSILON, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix

import torch
import torch.utils
from torch.utils.data.dataloader import DataLoader, default_collate


class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
        speaker_dict: dict = None,
    ):
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        self.speaker_dict = speaker_dict
        self.extractor = Fbank(FbankConfig(num_mel_bins=128))
        
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
        
    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_duration and max_cuts.
        """
        # validate_multi_kd(cuts)

        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        # import pdb; pdb.set_trace()
        try:
            input_tpl = self.input_strategy(cuts)
        except:
            cuts.to_jsonl("bad_cuts.jsonl.gz")
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)
        
        if self.speaker_dict is not None:
            labels = torch.tensor([self.speaker_dict[cut.supervisions[0].speaker] for cut in cuts])
            labels = labels.unsqueeze(1) # (N,1)
        else:
            labels = None

        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)
            
        dummy_text = "This is dummy text"
        batch = {
            "inputs": inputs,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text if supervision.text is not None else dummy_text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
            "labels": labels,
        }
        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        return batch
    
def speaker_id_to_label(speaker_ids, speaker_dict):
    """Generate one-hot speaker labels.

    Args:
        speaker_ids (_type_): _description_
        speaker_dict (_type_): _description_
    """
    N = len(speaker_ids)
    num_total_speakers = len(speaker_dict)
    ids = [speaker_dict[spk] for spk in speaker_ids]
    labels = torch.zeros(N, num_total_speakers).int()
    for i in range(N):
        labels[i, ids[i]] = 1
    return labels
    