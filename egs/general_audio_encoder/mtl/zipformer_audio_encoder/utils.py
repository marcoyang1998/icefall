import logging
import collections
import os
import re
from typing import List, Tuple
from lhotse.array import Array, TemporalArray

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

def remove_non_alphabetic(text: str, strict: bool = True) -> str:
    # Recommend to set strict to False
    if not strict:
        # Note, this also keeps space, single quote(')
        text = text.replace("-", " ")
        text = text.replace("—", " ")
        return re.sub(r"[^a-zA-Z0-9\s']+", "", text)
    else:
        # only keeps space
        return re.sub(r"[^a-zA-Z\s]+", "", text)


def upper_only_alpha(c):
    text = c.supervisions[0].text
    text = remove_non_alphabetic(text.upper(), strict=False)
    c.supervisions[0].text = text
    return c

def setup_distributed():
    """Setup distributed training environment."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank

def add_dummy_text(c):
    if c.supervisions[0].text is None:
        c.supervisions[0].text = "Dummy text added as a place holder. Please ignore this if possible."
    return c

def _add_dummy_embeddings_and_taskIDs(task_ID: int, c):
    whisper_embedding_dict = {
        'array': {'storage_type': 'numpy_hdf5', 'storage_path': 'data/dummy_embeddings/dummy_whisper_embedding_1510.h5', 'storage_key': 'dummy_whisper_embedding_1510', 'shape': [1510, 1280]}, 'temporal_dim': 0, 'frame_shift': 0.02, 'start': 0
    }
    whisper_dummy_embedding = TemporalArray.from_dict(whisper_embedding_dict)
    
    whisper_cb_indexes_dict = {
        'array': {'storage_type': 'numpy_hdf5', 'storage_path': 'data/dummy_embeddings/dummy_whisper_codebook_indexes_1510.h5', 'storage_key': 'dummy_whisper_codebook_indexes_1510', 'shape': [1510, 16]}, 'temporal_dim': 0, 'frame_shift': 0.02, 'start': 0
    }
    whisper_cb_indexes = TemporalArray.from_dict(whisper_cb_indexes_dict)
    
    beats_embedding_dict = {
        'storage_type': 'numpy_hdf5', 'storage_path': 'data/dummy_embeddings/dummy_beats_embedding.h5', 'storage_key': 'dummy_beats_embedding', 'shape': [527]
    }
    beats_dummy_embedding = Array.from_dict(beats_embedding_dict)
    
    ecapa_embedding_dict = {
        'storage_type': 'numpy_hdf5', 'storage_path': 'dummy_ecapa_embedding.h5', 'storage_key': 'dummy_ecapa_embedding', 'shape': [1, 192]
    }
    ecapa_dummy_embedding = Array.from_dict(ecapa_embedding_dict)
    
    mert_embedding_dict = {
        'array': {'storage_type': 'numpy_hdf5', 'storage_path': 'data/dummy_embeddings/dummy_mert_embedding_2260.h5', 'storage_key': 'dummy_mert_embedding', 'shape': [2260, 1024]}, 'temporal_dim': 0, 'frame_shift': 0.013333333333333334, 'start': 0
    }
    mert_dummy_embedding = TemporalArray.from_dict(mert_embedding_dict)
    
    def add_embeddings(c):
        # if not c.has_custom("whisper_embedding"):
        #     c.whisper_embedding = whisper_dummy_embedding
        if not c.has_custom("codebook_indexes"):
            c.codebook_indexes = whisper_cb_indexes
        
        # if not c.has_custom("ecapa_embedding"):
        #     c.ecapa_embedding = ecapa_dummy_embedding
        if not c.has_custom("beats_embedding"):
            c.beats_embedding = beats_dummy_embedding
        # if not c.supervisions[0].has_custom("audio_event"):
        #     c.supervisions[0].audio_event = "0"
        if c.supervisions[0].text is None:
            c.supervisions[0].text = "Dummy text added as a place holder. Please ignore this if possible."
        if task_ID is not None:
            c.task_id = task_ID
        return c
    
    c = add_embeddings(c)
    return c

def _add_task_id(task_id, c):
    c.task_id = task_id
    return c

def compare_model(state_dict1, state_dict2):
    assert state_dict1.keys() == state_dict2.keys()
    for key in state_dict1.keys():
        if torch.all(state_dict1[key] == state_dict2[key]):
            logging.info(f"Param: {key} is the same as new state dict")
        else:
            logging.info(f"Param: {key} is updated from new state dict")

class MetricsTracker(collections.defaultdict):
    def __init__(self):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
        super(MetricsTracker, self).__init__(int)

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            if v - v == 0:
                ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    raise ValueError(f"Unexpected key: {k}")
        frames = "%.2f" % self["frames"]
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "%.2f" % self["utterances"]
            ans_utterances += "over " + str(utterances) + " utterances."

        return ans_frames + ans_utterances

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('ctc_loss', 0.1), ('att_loss', 0.07)]
        """
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            if ("audio_tagging" in k) or ("speaker_verification" in k):
                norm_value = (
                    float(v) / num_utterances
                )
            else:
                norm_value = (
                    float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
                )
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        """
        Reduce using torch.distributed, which I believe ensures that
        all processes get the total.
        """
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """Add logging information to a TensorBoard writer.

        Args:
            tb_writer: a TensorBoard writer
            prefix: a prefix for the name of the loss, e.g. "train/valid_",
                or "train/current_"
            batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)
