import math
import random
from threading import Lock
from typing import Callable, Dict, List, Optional, Union, Tuple

import torch
from torch.utils.data.dataloader import default_collate
import numpy as np

from lhotse import validate
from lhotse import Fbank, FbankConfig
from lhotse.cut import CutSet, MonoCut, Cut, MixedCut
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.dataset.collation import collate_matrices
from lhotse.utils import compute_num_frames, compute_num_samples, ifnone, LOG_EPSILON
from lhotse.workarounds import Hdf5MemoryIssueFix

from lhotse.cut.set import mix


def str2multihot(events: List[str], n_classes=527, id_mapping=None):
    # generate multi-hot class labels
    if not isinstance(events, list):
        events = [events]
    labels = [list(map(int, event.split(";"))) for event in events]
    batch_size = len(labels)
    out = torch.zeros(batch_size, n_classes)

    for i, label in enumerate(labels):
        if id_mapping is not None:
            label = [id_mapping[l] for l in label]
        out[i, label] = 1

    return out, labels


class MultiTaskKDDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the multi task speech and audio processing.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SimpleCutSampler` sampler.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': float tensor with shape determined by :attr:`input_strategy`:
                      - single-channel:
                        - features: (B, T, F)
                        - audio: (B, T)
                      - multi-channel: currently not supported
            'supervisions': [
                {
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S

                    # For feature input strategies
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)

                    # For audio input strategies
                    'start_sample': Tensor[int] of shape (S,)
                    'num_samples': Tensor[int] of shape (S,)

                    # Optionally, when return_cuts=True
                    'cut': List[AnyCut] of len S
                }
            ]
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``S`` - number of supervision segments (greater or equal to B, as each Cut may have multiple supervisions)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features

    The 'sequence_idx' field is the index of the Cut used to create the example in the Dataset.
    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
        target_frame_rate: int = 50,
        at_KD: bool = False,
        sv_KD: bool = False,
        enable_cache: bool = True,
        token_mixing: bool = False
    ):
        """
        IterableDataset constructor.

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
        :param enable_cache: Enables a cache for the codebook indexes
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        self.extractor = Fbank(FbankConfig(num_mel_bins=128))
        
        self.at_KD = at_KD
        self.sv_KD = sv_KD
        
        self.target_frame_rate = target_frame_rate
        self.dummy_codebook_indexes = torch.ones(1510, 16) * (-100)
        self.dummy_audio_logits = torch.ones(527) * 0.5
        
        self.enable_cache = enable_cache

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
        
        self.token_mixing = token_mixing

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
        inputs, input_lens, mix_gains, replacement_prob = self.load_audio_and_compute_fbank(cuts)

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)
        
        # MVQ tokens
        cuts_pre_mixed = [c if isinstance(c, MonoCut) else c.tracks[0].cut for c in cuts]
        cuts_pre_mixed = fix_start(cuts_pre_mixed)
        
        mvq_tokens, mvq_token_lens = _collate_custom_field(
            cuts_pre_mixed,
            "codebook_indexes",
            dummy=self.dummy_codebook_indexes,
            temporal_array=True,
            target_frame_rate=self.target_frame_rate,
            pad_value=-100,
        )
        
        # perform token mixing
        if self.token_mixing:
            mix_speech_mvq_tokens = self.prepare_mixed_mvq_tokens(mvq_tokens, cuts, field_name="codebook_indexes")
        else:
            mix_speech_mvq_tokens = mvq_tokens
        
        if self.at_KD:
            # at_targets = collate_custom_field(
            #     cuts_pre_mixed, "beats_embedding", pad_value=-100
            # ) # (N,C)
            at_targets = _collate_custom_field(
                cuts_pre_mixed, "beats_embedding", dummy=self.dummy_audio_logits, temporal_array=False
            ) # (N,C)
        else:        
            audio_events = [getattr(c.supervisions[0], "audio_event", "0") for c in cuts_pre_mixed] # the label indices are in CED format
            # at_targets, _ = str2multihot(audio_events) # (N, num_events)
            at_targets = None
            
        sv_targets = None
        
        # task ids
        task_ids = [c.task_id for c in cuts_pre_mixed]
        task_ids = torch.tensor(task_ids)
        
        dummy_text = "This is dummy text."
        
        batch = {
            "inputs": inputs,
            "cb_indexes": mvq_tokens,
            "cb_indexes_len": mvq_token_lens,
            "mixed_cb_indexes": mix_speech_mvq_tokens,
            "mixed_cb_indexes_len": mvq_token_lens,
            "replacement_probs": torch.tensor(replacement_prob),
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text if supervision.text is not None else dummy_text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
            "task_ids": task_ids,
            "at_targets": at_targets,
            "sv_targets": sv_targets,
        }
        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        return batch
    
    def load_audio_and_compute_fbank(self, cuts: CutSet):
        audios = []
        mix_ratios = []
        replacement_probs = []
        for cut in cuts:
            if isinstance(cut, MixedCut):
                if cut.mix_type == "noise":
                    audio = cut.load_audio()
                    mix_ratio = 0.0
                    replacement_prob = 0.0
                elif cut.mix_type == "speech":
                    audio, mix_ratio, replacement_prob = _load_mixed_cut_single(cut)
                else:
                    raise ValueError()
            else:
                audio = cut.load_audio()
                mix_ratio = 0.0
                replacement_prob = 0.0
            audios.append(audio)
            mix_ratios.append(mix_ratio)
            replacement_probs.append(replacement_prob)
        
        inputs, input_lens = compute_feature(audios, cuts, self.extractor)
        
        return inputs, input_lens, mix_ratios, replacement_probs
    
    def prepare_mixed_mvq_tokens(
        self,
        mvq_tokens: torch.Tensor,
        cuts: CutSet,
        field_name: str = "codebook_indexes"
    ):
        mvq_tokens_mix = []
        
        for i,c in enumerate(cuts):
            if not isinstance(c, MixedCut):
                mvq_tokens_mix.append(mvq_tokens[i])
                continue
            if c.mix_type != "speech":
                mvq_tokens_mix.append(mvq_tokens[i])
                continue
            cur_mvq_tokens = mvq_tokens[i].clone()
            orig_track, mix_track = c.tracks # get the two tracks
            
            # compute the starting mixing frame
            offset = int(mix_track.offset * self.target_frame_rate)
            mixed_in_cb = torch.from_numpy(mix_track.cut.load_custom(field_name))
            mix_length = mixed_in_cb.shape[0]
            if mix_length + offset >= mvq_tokens.size(1):
                mix_length = mvq_tokens.size(1) - offset
                mixed_in_cb = mixed_in_cb[:, :mix_length]
            
            # replace the codebook indexes in the mix region
            cur_mvq_tokens[offset:offset + mix_length, :] = mixed_in_cb
            mvq_tokens_mix.append(cur_mvq_tokens)
        
        mvq_tokens_mix = torch.stack(mvq_tokens_mix, dim=0)
        return mvq_tokens_mix
            
    
    def mix_mvq_tokens(
        self,
        mvq_tokens: torch.Tensor,
        cuts: CutSet,
        mix_probs: List[float],
        field_name: str = "codebook_indexes"
    ):
        # Randomly replace a proportion of the original codebook indexes
        # with the codebook indexes from the mixed cut. The proportion is determined
        # by the gain of the mixed audio
        for i,c in enumerate(cuts):
            if not isinstance(c, MixedCut):
                continue
            if c.mix_type != "speech":
                continue
            orig_track, mix_track = c.tracks # get the two tracks
            
            # compute the starting mixing frame
            offset = int(mix_track.offset * self.target_frame_rate) 
            mixed_in_cb = torch.from_numpy(mix_track.cut.load_custom(field_name)) # should be only within the mix region
            mix_length = mixed_in_cb.shape[0]
            if mix_length + offset >= mvq_tokens.size(1):
                mix_length = mvq_tokens.size(1) - offset
                mixed_in_cb = mixed_in_cb[:, :mix_length]
            cur_cb_slice = mvq_tokens[i, offset:offset + mix_length, :]
            mixed_cb = _mix_tokens_single(cur_cb_slice, mixed_in_cb, mix_probs[i])
            mvq_tokens[i, offset:offset + mix_length] = mixed_cb
        return mvq_tokens

def compute_prob_from_energies(gain: float, energy_main: float, energy_mixin: float) -> float:
    """
    Computes the probability of selecting the mix-in token based on the 
    actual energy contribution of the mix-in signal to the total mixture.
    
    Args:
        gain: The linear gain applied to the mix-in signal (amplitude).
        energy_main: The energy (mean squared) of the main signal (A).
        energy_mixin: The energy (mean squared) of the mix-in signal (B) BEFORE gain.
        
    Returns:
        float: Probability between 0.0 and 1.0 representing the mix-in's dominance.
    """
    # 1. Calculate the energy of the mix-in signal AFTER the gain is applied.
    # Since Energy ~ Amplitude^2, we square the gain.
    energy_mixin_effective = (gain ** 2) * energy_mixin
    
    # 2. Compute total energy (assuming signals are uncorrelated).
    energy_total = energy_main + energy_mixin_effective
    
    # 3. Compute the ratio of the mix-in energy to the total energy.
    # Add epsilon to prevent division by zero.
    prob = energy_mixin_effective / (energy_total + 1e-8)
    
    return float(prob)

def audio_energy(audio: np.ndarray):
    return float(np.average(audio**2))

def _load_mixed_cut_single(cut: MixedCut):
    """
    Loads a mixed cut, mixing the second track (noise/interference) into the 
    first track (signal) at a specific SNR, and computes the mixing probability.
    """
    assert len(cut.tracks) == 2, "Only support mixing two cuts (Signal + Interference)"
    
    # 1. Load components
    sample_rate = cut.sampling_rate
    track_sig = cut.tracks[0]
    track_noise = cut.tracks[1]
    
    # Load full audio for both
    # We use .load_audio() on the underlying cut to get the raw samples
    orig_audio = track_sig.cut.load_audio()
    mix_in_audio = track_noise.cut.load_audio()
    
    # 2. Calculate Offsets using Lhotse precision tools
    # Offset of the noise relative to the start of the signal
    offset_seconds = track_noise.offset - track_sig.offset
    mix_offset_frames = compute_num_samples(offset_seconds, sample_rate)
    
    mix_in_frames = mix_in_audio.shape[1]
    sig_frames = orig_audio.shape[1]

    # 3. Determine Overlap Region in Signal
    start_sample = max(0, mix_offset_frames)
    end_sample = min(sig_frames, mix_offset_frames + mix_in_frames)
    
    # Slice the signal where the overlap occurs
    sig_slice = orig_audio[0, start_sample:end_sample]
    
    # 4. Calculate Energies
    energy_sig_local = audio_energy(sig_slice)
    energy_mix_in = audio_energy(mix_in_audio)
    
    # --- FIX: THE SILENCE BUG ---
    # If the local slice is silent, gain calculation will fail (gain=0).
    # We fallback to the global energy of the signal to ensure the noise is still added.
    if energy_sig_local < 1e-9:
        # Fallback to global energy of the utterance
        energy_sig_local = audio_energy(orig_audio) + 1e-9

    # 5. Calculate Gain
    # SNR definition: SNR = 10 * log10(E_signal / E_noise)
    # Therefore: E_target_noise = E_signal * 10^(-SNR/10)
    target_noise_energy = energy_sig_local * (10.0 ** (-track_noise.snr / 10))
    
    # Gain = sqrt(Target / Current)
    gain = math.sqrt(target_noise_energy / (energy_mix_in + 1e-9))
    
    # 6. Calculate Replacement Probability (for Tokens)
    replacement_prob = compute_prob_from_energies(gain, energy_sig_local, energy_mix_in)
    
    # 7. Mix Audio in-place
    # Determine the slice of the noise audio to add 
    noise_start_idx = 0 if mix_offset_frames >= 0 else abs(mix_offset_frames)
    noise_end_idx = noise_start_idx + (end_sample - start_sample)
    
    if noise_end_idx > noise_start_idx: # Ensure we have valid frames to mix
        orig_audio[0, start_sample:end_sample] += gain * mix_in_audio[0, noise_start_idx:noise_end_idx]
    
    return orig_audio, gain, replacement_prob
        
def mix_audio_with_offset(
    reference_cut: Cut,
    mixed_in_cut: Cut,
    snr: float = 10.0,
    drop_mixed_in_supervision: bool = True
):
    if drop_mixed_in_supervision:
        mixed_in_cut = mixed_in_cut.drop_supervisions()
    ref_duration = reference_cut.duration
    mixed_in_duration = mixed_in_cut.duration
    
    mix_duration = random.uniform(0, ref_duration / 2)
    
    # randomly truncate the mixed_in_cut to mix_duration if longer
    if mixed_in_duration > mix_duration:
        diff = mixed_in_duration - mix_duration
        truncate_start = random.uniform(0, diff)
        mixed_in_cut = mixed_in_cut.truncate(offset=truncate_start, duration=mix_duration)
        
    actual_mix_duration = min(mixed_in_cut.duration, mix_duration)
    offset = random.uniform(0, ref_duration - actual_mix_duration - 0.05) # a tolerance of 0.05 for safety
    mixed_cut = mix(
        reference_cut=reference_cut,
        mixed_in_cut=mixed_in_cut,
        offset=offset,
        snr=snr,
        preserve_id="left",
    )
    
    return mixed_cut

def _mix_tokens_single(A: torch.Tensor, B: torch.Tensor, p: float) -> torch.Tensor:
    """
    从 A 中随机选出 p% 的位置，用 B 中对应位置的值替换。
    
    参数:
        A (Tensor): 原始张量，形状为 (T, C)
        B (Tensor): 替换来源张量，形状必须与 A 相同
        p (float): 替换比例，范围为 0~1

    返回:
        Tensor: 替换后的新张量
    """
    assert A.shape == B.shape, "A and B must have the same shape"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    
    # 创建一个与 A 相同形状的 mask，表示哪些位置需要替换
    mask = torch.rand_like(A, dtype=torch.float32) < p

    # 创建新的张量：如果 mask 为 True，就用 B 的值，否则用 A 的值
    return torch.where(mask, B, A)

def compute_feature(audios, cuts, extractor):
    # compute features given the audios
    # cuts is only for metadata reading
    features_single = []
    for idx, (audio, cut) in enumerate(zip(audios, cuts)):
        try:
            features = extractor.extract(audio, cuts[idx].sampling_rate)
        except:
            print(
                f"Error while extracting the features for cut with ID {cut.id} -- details:\n{cut}"
            )
            raise
        features_single.append(torch.from_numpy(features))
    
    features_batch = collate_matrices(features_single, padding_value=LOG_EPSILON)
    
    feature_lens = torch.tensor(
        [f.shape[0] for f in features_single], dtype=torch.int64
    )

    out = (features_batch, feature_lens)
    return out

def gain2prob(gain: float, alpha: float=2.0):
    # x**alpha/(1+x**alpha), x is gain, alpha is empirically tuned
    return gain ** alpha / (1 + gain**alpha)

def fix_start(cuts):
    # make the start of codebook indexes the same as the cut
    new_cuts = []
    for cut in cuts:
        if cut.has_custom("codebook_indexes") and (not isinstance(cut.codebook_indexes, dict)):
            cut.codebook_indexes.start = cut.start
        if cut.has_custom("firered_codebook_indexes") and (not isinstance(cut.firered_codebook_indexes, dict)):
            cut.firered_codebook_indexes.start = cut.start
        new_cuts.append(cut)
    return new_cuts

def validate_multi_kd(cuts: CutSet) -> None:
    for cut in cuts:
        # assert cut.has_features, cut
        assert cut.has_custom("task_id")
        if cut.task_id == 1: 
            # speech cuts, should have codebook indexes
            assert cut.codebook_indexes.array.storage_key != "dummy_whisper_codebook_indexes_1510"
        elif cut.task_id == 2:
            # audio cuts, should have audio logits
            assert cut.beats_embedding.storage_key != "dummy_beats_embedding"

def load_codebook_indexes(c):
    info = c.codebook_indexes
    
    if isinstance(info, dict):
        filename = info["path"]
        with open(filename, "rb") as f:
            cb_indexes = np.load(f)
        # return np.load(filename, mmap_mode="r")
    else:
        cb_indexes = c.load_custom("codebook_indexes")
    return cb_indexes
    
def _collate_custom_field(
    cuts: CutSet, 
    field: str,
    dummy: torch.Tensor = None,
    temporal_array: bool = True,
    target_frame_rate: int = 50,
    pad_value=None,
):
    
    # by default, we assert the frame_shift is 0.02
    if temporal_array:
        max_frames = [int(c.duration * target_frame_rate) for c in cuts]
        
        temporal_dim = 0
        pad_value = -100
        arrs = [
            torch.from_numpy(load_codebook_indexes(c)) if c.has_custom(field) else dummy for c in cuts # load the numpy codebook indexes
        ]
        for i, arr in enumerate(arrs):
            arrs[i] = arr[:max_frames[i],:]
        
        arr_lens = torch.tensor(
            [a.shape[temporal_dim] for a in arrs], dtype=torch.int32
        )   
        largest_arr = max(arrs, key=torch.numel)
        maxlen = largest_arr.shape[temporal_dim]
        collated_shape = (len(arrs), *largest_arr.shape)
        dtype = largest_arr.dtype
        if any(d == dtype for d in (torch.uint8, torch.int8, torch.int16, torch.int32)):
            dtype = torch.int64
        tensors = pad_value * torch.ones(collated_shape, dtype=dtype)
        for aidx, a in enumerate(arrs):
            alen = a.shape[temporal_dim]
            # Construct an index expression such as tensors[:, :alen, :, :] programmatically;
            # All indices are set to ':', besides temporal dim which is determined on pad_direction.
            
            temporal_slice = slice(0, alen)
            indices = (aidx,) + tuple(
                temporal_slice if i == temporal_dim else slice(None, None, None)
                for i in range(len(a.shape))
            )
            tensors[indices] = a

        return tensors, arr_lens
    else:
        all_arrays = [torch.from_numpy(c.load_custom(field)) if c.has_custom(field) else dummy for c in cuts]
        return torch.stack(all_arrays)

def _test_mix():
    from lhotse import load_manifest_lazy
    manifest = "data/fbank/librispeech_cuts_dev-other.jsonl.gz"
    cuts = load_manifest_lazy(manifest).drop_features()
    reference_cut = cuts[0]
    noise_cuts = [cuts[4], cuts[2]] 
    
    for noise_cut in noise_cuts:
        mixed_cut = mix_audio_with_offset(reference_cut=reference_cut, mixed_in_cut=noise_cut, snr=5)
    print(mixed_cut)

if __name__=="__main__":
    from functools import partial
    from utils import _add_dummy_embeddings_and_taskIDs
    from lhotse import load_manifest
    
    _test_mix()
    
    # enable the cache
    
    dummy_codebook_indexes = torch.ones(1510, 16) * (-100)
    dummy_audio_logits = torch.ones(527) * 0.5
    
    cuts = load_manifest("data/vq_hubert_large_layer_21_normalize_1_cb_16/librispeech_cuts_dev-clean.jsonl.gz").subset(first=500).repeat(2)
    # cut_ids = [c.task_id for c in cuts]
    
    augmented_cuts = cuts.map(partial(_add_dummy_embeddings_and_taskIDs, None))
    # cuts = load_manifest("debug.jsonl.gz")
    
    # gt_mvq_tokens, gt_mvq_token_lens = collate_custom_field(augmented_cuts, "codebook_indexes", pad_value=-100)
    import time
    start = time.time()
    mvq_tokens, mvq_token_lens = _collate_custom_field(
        cuts,
        "codebook_indexes",
        dummy=dummy_codebook_indexes,
        temporal_array=True,
        pad_value=-100
    )
    # print(gt_mvq_tokens)
    
    # gt_beats_embed = collate_custom_field(augmented_cuts, "beats_embedding")
    # beats_embed = _collate_custom_field(cuts, "beats_embedding", dummy=dummy_audio_logits, temporal_array=False)
    
    # print(beats_embed)

