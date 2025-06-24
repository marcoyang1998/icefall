#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
# Copyright    2024  University of Cambridge        (authors: Xiaoyu Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
"""
Usage:

export CUDA_VISIBLE_DEVICES="0"

./zipformer/evaluate.py \
  --epoch 50 \
  --avg 10 \
  --exp-dir zipformer/exp \
  --max-duration 1000


"""

import argparse
from functools import partial
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from lhotse import CutSet
from kd_datamodule3_shar import MultiTaskDataModule

try:
    from sklearn.metrics import average_precision_score
except:
    raise ImportError(f"Please run\n" "pip3 install -U scikit-learn")
from train_multi_KD3_shar import add_model_arguments, get_model, get_params


from icefall.utils import AttributeDict, str2bool
from utils import _add_dummy_embeddings_and_taskIDs, _add_task_id


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )
    
    parser.add_argument(
        "--use-s3-client",
        type=str2bool,
        default=True,
    )
    
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="brainllm:s3://yangxiaoyu",
    )

    add_model_arguments(parser)

    return parser


def inference_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
):
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3, feature.shape

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    label = batch["at_targets"]

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    cuts = supervisions["cut"]
    audio_events = [c.supervisions[0].audio_event for c in cuts]

    with torch.cuda.amp.autocast(enabled=True):
        encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)
        audio_logits = model.forward_audio_tagging(encoder_out, encoder_out_lens, return_logits=True)
    # convert to probabilities between 0-1
    audio_logits = audio_logits.sigmoid().detach().cpu()

    return audio_logits, label


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
) -> Dict:
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    all_logits = []
    all_labels = []

    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        num_cuts += len(cut_ids)

        audio_logits, labels = inference_one_batch(
            params=params,
            model=model,
            batch=batch,
        )

        all_logits.append(audio_logits)
        all_labels.append(labels)

        if batch_idx % 20 == 1:
            logging.info(f"Processed {num_cuts} cuts already.")
    logging.info("Finish collecting audio logits")

    return all_logits, all_labels


@torch.no_grad()
def main():
    parser = get_parser()
    MultiTaskDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    
    # ASR params
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    
    params.update(vars(args))
    
    logging.info("Evaluation started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        
    if params.use_s3_client:
        from petrel_client.client import Client
        conf_path = "/mnt/petrelfs/zhangchen/petreloss.conf"
        client = Client(conf_path)
        params.client = client
    else:
        params.client = None

    logging.info("About to create model")

    model = get_model(params)
    
    # import pdb; pdb.set_trace()
    checkpoint = torch.load(params.checkpoint)["model"]
    
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    args.return_cuts = True
    librispeech = MultiTaskDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts().subset(first=100)
    test_clean_cuts = test_clean_cuts.map(partial(_add_task_id, 1))
    test_dl = librispeech.test_dataloaders(test_clean_cuts)

    test_sets = ["test_clean_cuts"]

    for batch in test_dl:
        device = next(model.parameters()).device
        feature = batch["inputs"]
        assert feature.ndim == 3

        feature = feature.to(device)
        
        import pdb; pdb.set_trace()
        supervisions = batch["supervisions"]
        cuts = supervisions["cut"]
        cuts = CutSet.from_cuts(cuts)
        cuts.to_jsonl("test.jsonl.gz")
        feature_lens = supervisions["num_frames"].to(device)
        
        with torch.amp.autocast("cuda", enabled=True):
            encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)
            
        import pdb; pdb.set_trace()
        torch.save(feature, "fbank_gt.pt")
        torch.save(encoder_out, "encoder_out_gt.pt")
        torch.save(encoder_out_lens, "encoder_out_lens_gt.pt")
        
        break


    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
