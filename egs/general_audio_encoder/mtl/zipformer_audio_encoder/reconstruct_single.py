#!/usr/bin/env python3

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from lhotse import Fbank, FbankConfig

from train_multi_KD3_shar_w2v2_mask_with_recon import add_model_arguments, get_model, get_params

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


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="Checkpoint epoch to use when --iter <= 0",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="If positive, use iteration checkpoints and ignore --epoch",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--wav-list",
        type=str,
        required=True,
        help="Path to wav list file. Supports .txt (one path per line) and .json (list of paths).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save reconstructed spectrogram npy files",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output JSON mapping absolute wav path -> absolute reconstructed npy path",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for loading/resampling audio",
    )

    add_model_arguments(parser)

    return parser


def load_wav_list(wav_list_path: Path) -> List[str]:
    if not wav_list_path.exists():
        raise FileNotFoundError(f"wav list file does not exist: {wav_list_path}")

    if wav_list_path.suffix.lower() == ".json":
        with wav_list_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON wav list must be a list of file paths")
        wavs = [str(x).strip() for x in data if str(x).strip()]
    else:
        with wav_list_path.open("r", encoding="utf-8") as f:
            wavs = [line.strip() for line in f if line.strip()]

    if not wavs:
        raise ValueError(f"No wav files found in wav list: {wav_list_path}")

    return [str(Path(w).expanduser().resolve()) for w in wavs]


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)


def compute_fbank(
    audio: torch.Tensor,
    extractor,
    sample_rate: int,
):
    audio = audio.squeeze(0)
    feature = [extractor.extract(audio, sampling_rate=sample_rate)]
    feature_lens = [f.size(0) for f in feature]

    feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True, padding_value=LOG_EPS)
    feature_lens = torch.tensor(feature_lens)
    return feature, feature_lens


def reconstruct_feature(
    model: nn.Module,
    extractor,
    audio: torch.Tensor,
    sample_rate: int,
    reconstruction_target_type: str = "mel",
    reconstruction_target_dim: int = 128,
):
    device = next(model.parameters()).device

    feature, feature_lens = compute_fbank(audio, extractor, sample_rate=sample_rate)
    feature = feature.to(device)
    feature_lens = feature_lens.to(device)

    encoder_out, _ = model.forward_encoder(feature, feature_lens)
    n = encoder_out.shape[0]
    pred = model.reconstruction_proj(encoder_out)
    pred = pred.reshape(n, -1, reconstruction_target_dim)

    if reconstruction_target_type == "mel":
        target = (feature - FBANK_MEAN) / math.sqrt(FBANK_VAR)
    elif reconstruction_target_type == "linear":
        audio = audio.to(device)
        target = model.reconstruction_target_extractor(audio)
        target = (target - LINEAR_SPECTROGRAM_MEAN) / math.sqrt(LINEAR_SPECTROGRAM_VAR)
    else:
        raise ValueError(f"Unsupported reconstruction target: {reconstruction_target_type}")

    return pred, target


def load_model(params: AttributeDict, device: torch.device) -> nn.Module:
    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[: params.avg]
            if len(filenames) == 0:
                raise ValueError(f"No checkpoints found for --iter {params.iter}, --avg {params.avg}")
            if len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device), strict=False)
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
            model.to(device)
        else:
            start = params.epoch - params.avg + 1
            filenames = [f"{params.exp_dir}/epoch-{i}.pt" for i in range(start, params.epoch + 1) if i >= 1]
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device), strict=False)
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[: params.avg + 1]
            if len(filenames) == 0:
                raise ValueError(f"No checkpoints found for --iter {params.iter}, --avg {params.avg}")
            if len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for --iter {params.iter}, --avg {params.avg}"
                )

            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating averaged model over iteration checkpoints "
                f"from {filename_start} (excluded) to {filename_end}"
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
                f"Calculating averaged model over epoch range from {start} (excluded) to {params.epoch}"
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

    model.eval()
    return model


@torch.no_grad()
def extract_reconstructions(
    model: nn.Module,
    wav_files: List[str],
    params: AttributeDict,
) -> Dict[str, str]:
    params.output_dir.mkdir(parents=True, exist_ok=True)
    fbank_extractor = Fbank(FbankConfig(num_mel_bins=128))

    mapping: Dict[str, str] = {}

    for i, wav_path in enumerate(wav_files):
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != params.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, params.sample_rate)

        pred, _ = reconstruct_feature(
            model=model,
            extractor=fbank_extractor,
            audio=waveform,
            sample_rate=params.sample_rate,
            reconstruction_target_type=params.reconstruction_target,
            reconstruction_target_dim=params.reconstruction_target_dim,
        )

        pred_np = pred.squeeze(0).detach().cpu().numpy()
        abs_wav_path = str(Path(wav_path).expanduser().resolve())
        npy_name = f"{i:08d}_{_sanitize_filename(Path(wav_path).stem)}.npy"
        npy_path = (params.output_dir / npy_name).resolve()
        np.save(npy_path, pred_np)

        mapping[abs_wav_path] = str(npy_path)

        if (i + 1) % 50 == 0:
            logging.info(f"Processed {i + 1}/{len(wav_files)} wav files")

    return mapping


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))
    params.exp_dir = Path(params.exp_dir)
    params.wav_list = Path(params.wav_list)
    params.output_dir = Path(params.output_dir)
    params.output_json = Path(params.output_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(params)

    model = load_model(params, device)

    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of model parameters: {num_param}")

    wav_files = load_wav_list(params.wav_list)
    logging.info(f"Loaded {len(wav_files)} wav files from {params.wav_list}")

    mapping = extract_reconstructions(model=model, wav_files=wav_files, params=params)

    params.output_json.parent.mkdir(parents=True, exist_ok=True)
    with params.output_json.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved wav->reconstruction mapping to {params.output_json}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
