import math
import random
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
        audio, audio_lens = read_audio(cuts)
        inputs, input_lens, mix_gains, replacement_prob = self.load_audio_and_compute_fbank(cuts)

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        
        # Fix the duration of the supervision after mixing
        cuts = cuts.map(fix_supervision_duration)
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
            mvq_tokens, mvq_token_lens = self.mix_mvq_tokens(
                mvq_tokens,
                mvq_token_lens,
                cuts,
                replacement_prob,
            )
        
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
            "audio": audio,
            "audio_lens": audio_lens,
            "cb_indexes": mvq_tokens,
            "cb_indexes_len": mvq_token_lens,
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
                mix_type = getattr(cut, "mix_type", "noise")
                if mix_type == "speech":
                    # audio, mix_ratio, replacement_prob = _load_mixed_cut_single(cut)
                    audio, mix_ratio, replacement_prob = _load_mixed_cut_single2(cut)
                elif mix_type == "noise":
                    audio = cut.load_audio()
                    mix_ratio = 0.0
                    replacement_prob = 0.0
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
    
    def mix_mvq_tokens(
        self,
        mvq_tokens: torch.Tensor,
        mvq_token_lens: torch.Tensor,
        cuts: CutSet,
        mix_probs: List[float],
        field_name: str = "codebook_indexes"
    ):
        from torch.nn.utils.rnn import pad_sequence
        
        processed_tokens = []
        new_token_lens = []
        for i, c in enumerate(cuts):
            # Start with the original tokens for this cut, removing padding
            orig_len = mvq_token_lens[i]
            current_tokens = mvq_tokens[i, :orig_len]

            if isinstance(c, MixedCut) and c.mix_type == "speech":
                orig_track, mix_track = c.tracks
                
                # Load the codebook for the mixed-in track
                mixed_in_cb = torch.from_numpy(mix_track.cut.load_custom(field_name)).to(current_tokens.device)
                mix_length = mixed_in_cb.shape[0]
                
                # Compute the mixing region
                offset = int(mix_track.offset * self.target_frame_rate)
                
                # Determine the required length after mixing
                required_length = offset + mix_length
                
                # Pad the original tokens if they are shorter than the mixed result
                if required_length > current_tokens.shape[0]:
                    pad_length = required_length - current_tokens.shape[0]
                    padding = torch.full((pad_length, current_tokens.shape[1]), 
                                         -100, dtype=current_tokens.dtype, device=current_tokens.device)
                    current_tokens = torch.cat([current_tokens, padding], dim=0)

                # Mix the overlapping region
                overlap_end = min(offset + mix_length, orig_len)
                overlap_length = overlap_end - offset
                
                if overlap_length > 0:
                    mixed_in_cb_overlap = mixed_in_cb[:overlap_length, :]
                    cur_cb_slice = current_tokens[offset:overlap_end, :]
                    mixed_cb = _mix_tokens_single(cur_cb_slice, mixed_in_cb_overlap, mix_probs[i])
                    current_tokens[offset:overlap_end] = mixed_cb

                # Handle the part of the mixed-in audio that extends beyond the original
                if offset + mix_length > orig_len:
                    # This part of the mixed_in_cb goes into the padded area of current_tokens
                    remaining_start_in_mixed_cb = max(0, orig_len - offset)
                    remaining_tokens = mixed_in_cb[remaining_start_in_mixed_cb:]
                    
                    # Determine where to place these remaining tokens in the target
                    paste_start_in_current = offset + remaining_start_in_mixed_cb
                    paste_end_in_current = paste_start_in_current + remaining_tokens.shape[0]

                    if remaining_tokens.shape[0] > 0:
                        current_tokens[paste_start_in_current:paste_end_in_current] = remaining_tokens

            processed_tokens.append(current_tokens)
            new_token_lens.append(current_tokens.shape[0])

        # Pad all processed tensors to the same length
        padded_tokens = pad_sequence(processed_tokens, batch_first=True, padding_value=-100)
        
        return padded_tokens, torch.tensor(new_token_lens, dtype=torch.int32)
    
    def _mix_mvq_tokens_deprecated(
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

def fix_supervision_duration(c):
    # after the mixing, the cut may become longer, which causes a mismatch
    # between the cut duration and the supervision duration.
    # Therefore, we modify the duration of the supervision to match the cut duration.
    if isinstance(c, MixedCut):
        if c.supervisions[0].duration != c.duration:
            sup = c.tracks[0].cut.supervisions[0]
            sup.duration = c.duration
            c.tracks[0].cut.supervisions = [sup]
    return c

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

def read_audio(cuts: CutSet):
    audios = []
    audio_lens = []
    for cut in cuts:
        audio = torch.from_numpy(cut.load_audio())
        audio_len = audio.shape[1]
        audios.append(audio[0])
        audio_lens.append(audio_len)
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    audio_lens = torch.tensor(audio_lens)
    return audios, audio_lens


def audio_energy(audio: np.ndarray):
    # return the average energy of the audio
    return float(np.average(audio**2))

def _load_mixed_cut_single2(cut: MixedCut):
    # ---------------------------------------------------------
    # 1. Load Components & Safety Checks
    # ---------------------------------------------------------
    sample_rate = cut.sampling_rate
    track_sig = cut.tracks[0]
    track_noise = cut.tracks[1]
    
    # 强制转换为 float32 以避免 int16 溢出或精度丢失
    orig_audio = track_sig.cut.load_audio()
    mix_in_audio = track_noise.cut.load_audio()
    
    # get the power and compute the gain
    power_sig = audio_energy(orig_audio)
    power_noise = audio_energy(mix_in_audio)
    target_noise_power = power_sig * (10.0 ** (-track_noise.snr / 10))
    
    if power_noise > 1e-12:
        # Gain = sqrt(Target / Source)
        gain = math.sqrt(target_noise_power / power_noise)
    else:
        # 噪音文件是静音
        gain = 0.0
        
    # compute the replacement probability
    replacement_prob = compute_prob_from_energies(gain, power_sig, power_noise)
    
    # compute the resulting audio
    result_audio = cut.load_audio()
    
    return result_audio, gain, replacement_prob

def _load_mixed_cut_single(cut: MixedCut):
    """
    Loads a mixed cut (Signal + Interference).
    
    Improvements:
    1. Uses Global Power (Mean Square) for stable SNR calculation.
    2. Uses float32 to prevent overflow during mixing.
    3. Handles variable lengths correctly.
    """
    assert len(cut.tracks) == 2, "Only support mixing two cuts (Signal + Interference)"
    
    # ---------------------------------------------------------
    # 1. Load Components & Safety Checks
    # ---------------------------------------------------------
    sample_rate = cut.sampling_rate
    track_sig = cut.tracks[0]
    track_noise = cut.tracks[1]
    
    # 强制转换为 float32 以避免 int16 溢出或精度丢失
    orig_audio = track_sig.cut.load_audio().astype(np.float32)
    mix_in_audio = track_noise.cut.load_audio().astype(np.float32)
    
    # 获取维度信息
    num_channels = orig_audio.shape[0]
    sig_frames = orig_audio.shape[1]
    mix_in_frames = mix_in_audio.shape[1]

    # ---------------------------------------------------------
    # 2. Calculate Offsets & Dimensions
    # ---------------------------------------------------------
    # 计算噪音相对于信号的偏移量
    offset_seconds = track_noise.offset - track_sig.offset
    mix_offset_frames = compute_num_samples(offset_seconds, sample_rate)
    
    # 计算总时长：因为你提到第二个音频永远在第一个之后(或中间)，
    # 所以总长度是 max(信号长, 偏移量 + 噪音长)
    total_frames = max(sig_frames, mix_offset_frames + mix_in_frames)
    
    # 初始化输出音频
    if total_frames > sig_frames:
        # 需要扩展长度
        result_audio = np.zeros((num_channels, total_frames), dtype=np.float32)
        result_audio[:, :sig_frames] = orig_audio
    else:
        # 长度足够，直接拷贝副本
        result_audio = orig_audio.copy()

    # ---------------------------------------------------------
    # 3. Calculate Gain using GLOBAL POWER (Stability Fix)
    # ---------------------------------------------------------
    # 使用整段音频的平均功率计算 SNR，避免局部静音导致的 Gain 跳变
    power_sig = audio_energy(orig_audio)
    power_noise = audio_energy(mix_in_audio)
    
    # 计算目标噪音功率
    # Target Power = Signal Power * 10^(-SNR/10)
    target_noise_power = power_sig * (10.0 ** (-track_noise.snr / 10))
    
    if power_noise > 1e-12:
        # Gain = sqrt(Target / Source)
        gain = math.sqrt(target_noise_power / power_noise)
    else:
        # 噪音文件是静音
        gain = 0.0

    # ---------------------------------------------------------
    # 4. Calculate Replacement Probability (Local Context)
    # ---------------------------------------------------------
    # 虽然 Gain 是全局定的，但在计算 Mask 概率时，通常需要参考“重叠区域”的能量
    overlap_start = max(0, mix_offset_frames)
    overlap_end = min(sig_frames, mix_offset_frames + mix_in_frames)
    
    if overlap_end > overlap_start:
        sig_slice = orig_audio[:, overlap_start:overlap_end]
        energy_sig_local = audio_energy(sig_slice)
    else:
        # 没有重叠（噪音在信号结束后才开始）
        energy_sig_local = 0.0 # 或者根据业务逻辑处理
        
    # 注意：这里传入的是 mix_in_audio 的原始功率/能量
    # 具体的 compute_prob_from_energies 内部逻辑需要与 audio_energy 的定义(Power)匹配
    replacement_prob = compute_prob_from_energies(gain, energy_sig_local, power_noise)

    # ---------------------------------------------------------
    # 5. Mix Audio
    # ---------------------------------------------------------
    mix_start_sample = max(0, mix_offset_frames)
    
    # 计算需要混合的帧数
    noise_frames_to_mix = min(mix_in_frames, total_frames - mix_start_sample)
    
    if noise_frames_to_mix > 0 and gain > 0:
        # 执行混合：Result = Signal + Gain * Noise
        # 假设通道数匹配，或者利用 numpy 的广播机制
        result_audio[:, mix_start_sample:mix_start_sample + noise_frames_to_mix] += \
            gain * mix_in_audio[:, :noise_frames_to_mix]
            
    return result_audio, gain, replacement_prob
        
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

