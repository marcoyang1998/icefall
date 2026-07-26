#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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



import argparse
import logging
import math
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from lhotse import load_manifest_lazy
from lhotse.cut import MonoCut
import numpy as np
import torch
import torch.nn as nn

from train_multi_KD3_shar_flow import add_model_arguments, get_model, get_params

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import AttributeDict, str2bool

LOG_EPS = math.log(1e-10)

FBANK_MEAN = -4.835472374046821
FBANK_VAR = 20.61560462170407

LINEAR_SPECTROGRAM_MEAN = -19.486874059807775
LINEAR_SPECTROGRAM_VAR = 341.9980405542238

LS_AS_LINEAR_SPECTROGRAM_MEAN = -20.516397641408407
LS_AS_LINEAR_SPECTROGRAM_VAR = 331.076620822691

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
    
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="The manifest for reconstruction"
    )
    
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of inference steps for reconstruction"
    )
    
    parser.add_argument(
        "--inference-schedule",
        type=str,
        default=None,
        help="A string of float values separated by comma, specifying the inference schedule. If not provided, it will be automatically generated as a linear schedule from 1 to 0."
    )

    add_model_arguments(parser)

    return parser

def compute_fbank(
    audio: torch.Tensor,
    extractor,
    feature_type: str = "mel",
):
    if feature_type == "mel":
        audio = audio.squeeze()
        feature = [extractor.extract(audio, sampling_rate=16000)]
        feature_lens = [f.size(0) for f in feature]
        
        feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True, padding_value=LOG_EPS)
        feature_lens = torch.tensor(feature_lens)
    elif feature_type == "linear":
        assert audio.ndim == 2
        feature = extractor(audio)
        feature_lens = torch.tensor([f.size(0) for f in feature])
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    return feature, feature_lens

def reconstruct_feature(
    model: nn.Module,
    extractor,
    cut: MonoCut,
    reconstruction_target_type: str = "mel",
    reconstruction_target_dim: int = 128,
    inference_steps: int = 10,
    inference_schedule: List[float] = None
):
    # return both the prediction and the groundtruth
    device = next(model.parameters()).device
    audio = torch.from_numpy(cut.load_audio())
    # this is the input mel features
    feature, feature_lens = compute_fbank(
        audio,
        extractor,
        feature_type="mel",
    )
    feature = feature.to(device)
    feature_lens = feature_lens.to(device)
    
    encoder_out, encoder_out_lens, _ = model.forward_encoder(feature, feature_lens) # (B,T,C)
    
    # the predition if already converted back to the original scale
    pred, pred_lens = model.inference_flow(
        encoder_out,
        encoder_out_lens,
        K=inference_steps,
        schedule=inference_schedule,
    ) # (B,T,C)
    
    if reconstruction_target_type == "mel":
        target = feature
    elif reconstruction_target_type == "linear":
        target = model.flow_target_extractor(audio)
        # target = (target - LINEAR_SPECTROGRAM_MEAN) / math.sqrt(LINEAR_SPECTROGRAM_VAR)
    else:
        raise ValueError(f"Unsupported reconstruction target: {reconstruction_target_type}")
    
    return pred, target


def plot_reconstruction(pred, target, output_path):
    """
    Plot reconstructed and target fbank features.
    pred, target: (B, T, C) or (T, C) tensors
    """
    # Remove batch dimension if present
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if target.dim() == 3:
        target = target.squeeze(0)

    # Detach and convert to numpy
    pred = pred.cpu().numpy().T  # (C, T) for plotting
    target = target.cpu().numpy().T

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot original
    im1 = axes[0].imshow(target, aspect="auto", origin="lower", interpolation="none")
    axes[0].set_title("Original Fbank")
    axes[0].set_ylabel("Mel Bin")
    fig.colorbar(im1, ax=axes[0])

    # Plot reconstructed
    im2 = axes[1].imshow(pred, aspect="auto", origin="lower", interpolation="none")
    axes[1].set_title("Reconstructed Fbank")
    axes[1].set_ylabel("Mel Bin")
    axes[1].set_xlabel("Frames")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved plot to {output_path}")


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    logging.info("Decoding started")

    device = torch.device("cpu")

    logging.info(f"Device: {device}")
    logging.info(params)

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
            info = model.load_state_dict(average_checkpoints(filenames, device=device), strict=False)
            logging.info(f"Load info: {info}")
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
            model.load_state_dict(average_checkpoints(filenames, device=device))
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
            load_info = model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=True,
            )
            logging.info(load_info)
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
            )

    
    from lhotse import Fbank, FbankConfig    
    fbank_extractor = Fbank(FbankConfig(num_mel_bins=128))
    
    
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    
    cuts = load_manifest_lazy(params.manifest)
    
    def change_source(c):
        source = c.recording.sources[0].source
        source = source.replace(
            "download/",
            "download3/" # use local
        )
        c.recording.sources[0].source = source
        return c
    if "audioset" in params.manifest:
        cuts = cuts.map(change_source)
    
    recon_dir = params.exp_dir / "reconstruction"
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    if params.inference_schedule is not None:
        inference_schedule = [float(x) for x in params.inference_schedule.split(",")]
    else:
        inference_schedule = None
    
    for i, cut in enumerate(cuts):
        pred, target = reconstruct_feature(
            model,
            fbank_extractor,
            cut,
            reconstruction_target_type=params.flow_target,
            reconstruction_target_dim=params.flow_target_dim,
            inference_steps=params.num_inference_steps,
            inference_schedule=inference_schedule,
        )
        
        plot_path = recon_dir / f"reconstruct_{i}.png"
        pred_npy_path = recon_dir / f"reconstruct_{i}.npz"
        target_npy_path = recon_dir / f"reconstruct_{i}_target.npz"
        np.save(pred_npy_path, pred.cpu().numpy())
        np.save(target_npy_path, target.cpu().numpy())
        plot_reconstruction(pred, target, plot_path)
        
        if i == 5:
            break

    logging.info("Done!")

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
