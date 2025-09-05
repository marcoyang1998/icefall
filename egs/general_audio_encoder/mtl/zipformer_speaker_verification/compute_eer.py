#!/usr/bin/env python3
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

./multi_task/compute_eer.py \
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
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

from sv_datamodule import SVDataModule
from train_sv import add_model_arguments, get_model, get_params

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


def get_test_pairs(dataset_name: str):
    test_mapping = {
        "VoxCeleb1": "download/veri_test.txt",
        "VoxCeleb1-cleaned": "download/veri_test2.txt",
        "VoxCeleb1-H": "download/list_test_hard.txt",
        "VoxCeleb1-H-cleaned": "download/list_test_hard2.txt",
        "VoxCeleb1-E": "download/list_test_all.txt",
        "VoxCeleb1-E-cleaned": "download/list_test_all2.txt",
    }
    test_file = test_mapping[dataset_name]
    with open(test_file, 'r') as f:
        lines = f.readlines()
    
    testing_pairs = [line.split() for line in lines]
    return testing_pairs

def similarity(embed1, embed2, threshold=0.25):
    sim = F.cosine_similarity(embed1, embed2, dim=-1, eps=1e-6)
    return sim, sim > threshold

def inference_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    feature = batch["inputs"]
    assert feature.ndim == 3, feature.shape

    feature = feature.to(device).to(dtype)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)
    
    speaker_embeddings = model.compute_speaker_embedding(encoder_out, encoder_out_lens)
    assert not speaker_embeddings.isnan().any()
    
    return speaker_embeddings


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

    embedding_dict = {}
    
    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        num_cuts += len(cut_ids)

        speaker_embeddings = inference_one_batch(
            params=params,
            model=model,
            batch=batch,
        )

        for id, embedding in zip(cut_ids, speaker_embeddings):
            embedding_dict[id] = embedding.detach().cpu()
        
        if batch_idx % 20 == 1:
            logging.info(f"Processed {num_cuts} cuts already.")
    logging.info(f"Finish collecting speaker embeddings")
        
    return embedding_dict


@torch.no_grad()
def main():
    parser = get_parser()
    SVDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    
    # ASR params
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    
    params.update(vars(args))

    params.res_dir = params.exp_dir / "speaker_verification"

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Evaluation started")

    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

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
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=True
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
            model.load_state_dict(
                average_checkpoints(filenames), strict=True
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
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                ),
                strict=True,
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
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                ),
                strict=False,
            )
            
            running_var = model.sv_module.asp.tdnn.norm.norm.state_dict()["running_var"]
            model.sv_module.asp.tdnn.norm.norm.running_var = torch.where(running_var>0, running_var, 0.0)
            running_mean = model.sv_module.asp.tdnn.norm.norm.state_dict()["running_mean"]
            model.sv_module.asp.tdnn.norm.norm.running_mean = torch.where(running_mean>0, running_mean, 0.0)
            
    model.to(device)
    model.eval()
    
    for n,p in model.named_parameters():
        assert not torch.isnan(p).any(), (n,p)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    args.return_cuts = True
    voxceleb = SVDataModule(args)

    voxceleb1_cuts = voxceleb.voxceleb_test_cuts()
    vox1_dl = voxceleb.test_dataloaders(voxceleb1_cuts)

    test_sets = ["VoxCeleb1-cleaned"]

    embedding_dict = decode_dataset(
        dl=vox1_dl,
        params=params,
        model=model,
    )

    for test_set in test_sets:
        evaluate_embeddings(
            test_set=test_set,
            embedding_dict=embedding_dict,
        )

    logging.info("Done")

def evaluate_embeddings(test_set: str, embedding_dict: Dict):
    # Evaluate the embeddings
    # Iterate over the testing pairs and tune the threshold
    logging.info(f"-----------For testing set: {test_set}------------")
    fa = 0
    fr = 0
    testing_pairs = get_test_pairs(test_set) 
    logging.info(f"A total of {len(testing_pairs)} pairs.")
    
    scores = []
    labels = []
    
    for i, pair in enumerate(testing_pairs):
        label, spkr1, spkr2 = pair
        embed1 = embedding_dict[spkr1]
        embed2 = embedding_dict[spkr2]
        
        sim, prediction = similarity(embed1, embed2)
        scores.append(sim.item())
        labels.append(int(label))
    
    # EER is where fpr == fnr
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    fnr = 1 - tpr
    index = np.nanargmin(np.absolute((fnr - fpr)))
    EER = 0.5 * (fnr[index] + fpr[index])
    
    op_thres = thresholds[index]
    FAR = fpr[index]
    FRR = fnr[index]
    logging.info("Operating threshold for {}: {:.4f}, FAR: {:.4f}, FRR: {:.4f}, EER: {:.4f}".format(test_set, op_thres, FAR, FRR, EER))
    logging.info(f"Finished testing for {test_set}")

if __name__ == "__main__":
    main()
