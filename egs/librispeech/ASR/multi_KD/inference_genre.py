import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lhotse import load_manifest
from kd_datamodule import LibriSpeechKDDataModule

from train_genre_classification import add_model_arguments, get_model, get_params, genre2id
from sklearn import metrics
from utils import get_class_dict

from icefall import ContextGraph
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    make_pad_mask,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

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
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    cuts = supervisions["cut"]
    feature_lens = supervisions["num_frames"].to(device)
    genres = [c.supervisions[0].genre for c in cuts]
    labels = genre2id(genres)
    
    encoder_out, encoder_out_lens, middle_out = model.forward_encoder(feature, feature_lens, return_middle_out=True)
    
    prediction = model.forward_genre_classification(encoder_out, encoder_out_lens)
    prediction = prediction.softmax(dim=-1).argmax(-1)
    
    return prediction, labels
    
    
def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
) -> Dict:
    num_cuts = 0
    predictions = []
    labels = []

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"
        
    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        num_cuts += len(cut_ids)

        prediction, label = inference_one_batch(
            params=params,
            model=model,
            batch=batch,
        )
        predictions.append(prediction)
        labels.append(label)
        
        if batch_idx % 20 == 1:
            logging.info(f"Processed {num_cuts} cuts already.")
        
    return predictions, labels

        
@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechKDDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))
        
    params.res_dir = params.exp_dir / "inference_genre_classification"
    
    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
        
    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"
        
    if params.causal:
        params.suffix += f"-chunk-size-{params.chunk_size}-left-context-frames-{params.left_context_frames}"
        
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
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device), strict=False)
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
            model.load_state_dict(average_checkpoints(filenames, device=device), strict=False)
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
                )
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
    
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    args.return_cuts = True
    librispeech = LibriSpeechKDDataModule(args, device=device, evaluation=True)
    
    def add_dummy_text(c):
        if c.supervisions[0].text is None:
            c.supervisions[0].text = "This is just dummy text!"
        return c
    
    test_cuts = librispeech.gtzan_test_cuts()
    test_cuts = test_cuts.map(add_dummy_text)
    gtzan_test_dl = librispeech.test_dataloaders(test_cuts)

    dev_cuts = librispeech.gtzan_dev_cuts()
    dev_cuts = dev_cuts.map(add_dummy_text)
    gtzan_dev_dl = librispeech.test_dataloaders(dev_cuts)

    test_sets = ["dev", "test"]
    test_dls = [gtzan_dev_dl, gtzan_test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        predictions, labels = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
        )
        
        predictions = torch.cat(predictions).tolist()
        labels = torch.cat(labels).tolist()
        acc = metrics.accuracy_score(predictions, labels)
        
        logging.info(f"The accuracy on {test_set} is: {acc}")
        
    logging.info("Done!")


if __name__ == "__main__":
    main()