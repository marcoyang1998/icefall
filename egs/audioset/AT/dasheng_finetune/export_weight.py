#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
import logging
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from at_datamodule import AudioSetATDatamodule

try:
    from sklearn.metrics import average_precision_score
except:
    raise ImportError(f"Please run\n" "pip3 install -U scikit-learn")
from train import add_model_arguments, get_model, get_params, str2multihot

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import AttributeDict, setup_logger, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    add_model_arguments(parser)

    return parser


def inference_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
):
    device = next(model.parameters()).device
    
    audios = batch["audio"].to(device)
    audio_lens = batch["audio_lens"].to(device)
    
    # the label indices are in CED format (https://github.com/RicherMans/CED)
    audio_event = batch["audio_events"]

    label, _ = str2multihot(audio_event)
    label = label.detach().cpu()

    encoder_out = model.forward_encoder(audios)
    audio_logits = model.forward_audio_tagging(encoder_out)

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
        cut_ids = [cut.id for cut in batch["cuts"]]
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
    AudioSetATDatamodule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "inference_audio_tagging"

    logging.info("Evaluation started")

    logging.info(params)

    device = torch.device("cpu")

    logging.info("About to create model")

    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=False
            )
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=False
            )
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )

    model.to(device)
    model.eval()
    
    weights = model.layer_weight.softmax(-1).cpu()
    logging.info(f"Layer weights: {weights}")
    
    weight_file = f"data/weights/{params.model_version}.pt"
    if os.path.exists(weight_file):
        logging.info(f"Weight file exists already at {weight_file}, don't overwrite.")
    else:    
        logging.info(f"Saving the weights to {weight_file}")
        torch.save(weights, weight_file)

    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    main()