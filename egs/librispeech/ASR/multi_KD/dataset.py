#      Copyright      2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Callable, Dict, List, Optional, Union

from utils import get_class_dict

import numpy as np
import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.supervision import SupervisionSegment
from lhotse.utils import compute_num_frames, ifnone

from torch.utils.data.dataloader import DataLoader, default_collate


class MultiKDDataset(torch.utils.data.Dataset):
    """This is a dataset for Prompt ASR. It supports the following features:
    1. Select a tuple of (text, pre_text, style_text) randomly from a
    list of texts as supervisions.

    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
        beats: torch.nn.Module = None,
        ecapa: torch.nn.Module = None,
        whisper: torch.nn.Module = None,
    ):
        """
        Icefall MultiKD IterableDataset constructor. See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py
        for more details.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk.
        :param text_sampling_func: Sampling a text as transcription from a list of texts.
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        
        # The teacher models
        self.beats = beats
        self.ecapa = ecapa
        self.whisper = whisper
        
        self.beats_class_dict = get_class_dict()

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        # validate_for_asr(cuts)

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
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        if len(input_tpl) == 4:
            # This means we are returning the audios as well
            inputs, input_lens, audios, audio_lens = input_tpl
            assert len(audios) == inputs.shape[0]
        else:
            inputs, _ = input_tpl
        
        with torch.no_grad():
            if self.beats is not None:
                beats_embeddings = self.beats.get_embeddings(audio=audios, audio_lens=audio_lens)
                beats_embeddings = beats_embeddings.unsqueeze(1)
            else:
                beats_embeddings = torch.tensor(0.)
            
            if self.ecapa is not None:
                ecapa_embeddings = self.ecapa.get_embeddings(audio=audios, audio_lens=audio_lens)
            else:
                ecapa_embeddings = torch.tensor(0.)
                
            if self.whisper is not None:
                whisper_embeddings = self.whisper.get_embeddings(audio=audios, audio_lens=audio_lens) # (N,T,C)
            else:
                whisper_embeddings = torch.tensor(0.)
        
        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)

        batch = {
            "inputs": inputs,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
            "beats_embedding": beats_embeddings,
            "ecapa_embedding": ecapa_embeddings,
            "whisper_embedding": whisper_embeddings,
        }
        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        has_word_alignments = all(
            s.alignment is not None and "word" in s.alignment
            for c in cuts
            for s in c.supervisions
        )

        return batch


class SpeakerRecognitionDataset(torch.utils.data.Dataset):
    """This is a dataset for Prompt ASR. It supports the following features:
    1. Select a tuple of (text, pre_text, style_text) randomly from a
    list of texts as supervisions.

    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
        ecapa: torch.nn.Module = None,
    ):
        """
        Icefall MultiKD IterableDataset constructor. See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py
        for more details.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk.
        :param text_sampling_func: Sampling a text as transcription from a list of texts.
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy

        self.ecapa = ecapa

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """

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
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        if len(input_tpl) == 4:
            # This means we are returning the audios as well
            inputs, input_lens, audios, audio_lens = input_tpl
            assert len(audios) == inputs.shape[0]
        else:
            inputs, input_lens = input_tpl
            audios = None
            
        with torch.no_grad():
            if self.ecapa is not None and audios is not None:
                ecapa_embeddings = self.ecapa.get_embeddings(audio=audios, audio_lens=audio_lens)
            else:
                ecapa_embeddings = torch.tensor(0.)

        batch = {
            "inputs": inputs,
            "supervisions": {
                "num_frames": input_lens,
            },
            "ecapa_embeddings": ecapa_embeddings,
        }
        if self.return_cuts:
            batch["supervisions"]["cut"] = [c for c in cuts]

        return batch



def validate_for_asr(cuts: CutSet) -> None:
    validate(cuts)
    tol = 2e-3  # 1ms
    for cut in cuts:
        for supervision in cut.supervisions:
            assert supervision.start >= -tol, (
                f"Supervisions starting before the cut are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )

            # Supervision start time is relative to Cut ...
            # https://lhotse.readthedocs.io/en/v0.10_e/cuts.html
            #
            # 'supervision.end' is end of supervision inside the Cut
            assert supervision.end <= cut.duration + tol, (
                f"Supervisions ending after the cut "
                f"are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )


def get_substring(s: str, min_len: int = 40, max_len: int = 250) -> str:
    """A helper function that generates a random substring from a given string

    Args:
        s (str): Input string

    Returns:
        str: Returned substring
    """
    min_len = min(len(s), min_len)

    start = random.randint(0, len(s) - min_len)
    end = min(start + max_len, random.randint(start + min_len, len(s)))

    return s[start:end]