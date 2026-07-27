
import argparse
import json
import logging
import re
import math
from typing import List, Dict, Tuple
from pathlib import Path

from train_multi_KD3_shar import add_model_arguments, get_encoder_embed, get_encoder_model
from zipformer2 import Zipformer2

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.compliance.kaldi import fbank as torch_fbank

from icefall.utils import AttributeDict, setup_logger, make_pad_mask

LOG_EPS=math.log(1e-10)
SAMPLE_RATE=16000

class ZipformerModel(torch.nn.Module):
    def __init__(
        self, encoder_embed: torch.nn.Module, encoder: Zipformer2
    ):
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        
        self.encoder_dim = encoder.encoder_dim
    
    def _get_full_dim_output_impl(self, outputs: List[torch.Tensor], max_depth):
        output_dim = max(self.encoder_dim[:max_depth])
        output_pieces = [outputs[-1]]
        cur_dim = self.encoder_dim[max_depth - 1]
        
        for i in range(max_depth - 2, -1, -1):
            d = self.encoder_dim[i]
            if d > cur_dim:
                this_output = outputs[i]
                output_pieces.append(this_output[..., cur_dim:d])
                cur_dim = d
        assert cur_dim == output_dim
        return torch.cat(output_pieces, dim=-1)
    
    def _get_full_dim_output(self, outputs: List[torch.Tensor], max_depth: int):
        outputs = outputs[:max_depth]
        return self._get_full_dim_output_impl(outputs, max_depth=max_depth)
    
    def get_embeddings(self, batch, layer_idx: int = -1):
        device = next(self.parameters()).device
        x = batch["feature"].to(device)
        x_lens = batch["num_frames"].to(device)
        
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens, layer_results = self.encoder(
            x, x_lens, src_key_padding_mask, return_middle_out=True
        )
        
        if layer_idx == -1:
            feature = encoder_out.permute(1, 0, 2)    
        else:
            # the intermediate layers' feature are 50 Hz
            feature = self._get_full_dim_output(layer_results, layer_idx)
            # feature = layer_results[layer_idx-1] # index starts from 1
            feature = feature.permute(1, 0, 2)
            encoder_out_lens = x_lens
        
        return feature, encoder_out_lens


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--wav-list",
        type=str,
        required=True,
        help="Path to a wav list file. Supports .txt (one wav path per line) and .json (list of paths)."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings_npy",
        help="Directory to save per-file embeddings as .npy"
    )

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output JSON file that maps each input wav filename to saved .npy embedding path"
    )

    parser.add_argument(
        "--embedding-layer",
        type=int,
        default=-1,
        help="Which layer's representation should be extracted, index start from 1, i.e the 10-th layer requires"
        "--embedding-layer 10"
    )

    # zipformer related args
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--zipformer-version",
        type=str,
        default="300m",
    )
    
    parser.add_argument(
        "--frame-shift",
        type=float,
        default=0.02,
        help="Frame shift in seconds for fbank extraction"
    )

    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=128,
        help="Number of mel bins used in fbank extraction"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for audio loading/resampling"
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=128,
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


def extract_fbank(wav_path: str, sample_rate: int=SAMPLE_RATE) -> torch.Tensor:
    waveform, sr = torchaudio.load(wav_path)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True) # (1, num_samples)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    fbank, fbank_lens = compute_fbank(waveform, torch.tensor([waveform.size(1)]))
    fbank = fbank.squeeze(0) # (T, num_mel_bins)
    return fbank

def compute_fbank(
    wavs: torch.Tensor, wav_lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute fbank features
    Args:
        wavs (torch.Tensor): the mono-channel input waveform, (N, T)
        wav_lens (torch.Tensor): the length of each waveform in samples (N)
    Returns:
        The fbank features, and their lengths
    """
    assert wavs.ndim == 2, wavs.shape
    low_freq = 20.0
    high_freq=-400.0
    dither=0.0
    snip_egdes=False

    features = []
    for i, wav in enumerate(wavs):
        feat = torch_fbank(
            wav[:wav_lens[i]].unsqueeze(0),
            sample_frequency=16000, # this is fixed to 16000
            num_mel_bins=128,
            low_freq=low_freq,
            snip_edges=snip_egdes,
            high_freq=high_freq,
            dither=dither,
            energy_floor=1.0e-10,
        )
        features.append(feat)
    feat_len = torch.tensor([f.shape[0] for f in features]).to(wavs.device)
    features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS).to(wavs.device)
    return features, feat_len


@torch.no_grad()
def extract_embeddings(wav_files: List[str], params: AttributeDict) -> Dict[str, str]:
    log_dir = params.output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(log_dir / "log-zipformer-embeddings"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(params)
    model = ZipformerModel(
        encoder_embed=get_encoder_embed(params),
        encoder=get_encoder_model(params),
    )
    state_dict = torch.load(params.model_ckpt, map_location="cpu")["model"]
    load_info = model.load_state_dict(state_dict, strict=False)
    logging.info(load_info)

    model.to(device)
    model.eval()
    logging.info(f"Number of zipformer model params: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Successfully loaded zipformer model.")

    params.output_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, str] = {}
    for i, wav_path in enumerate(wav_files):
        fbank = extract_fbank(
            wav_path=wav_path,
            sample_rate=params.sample_rate,
        )

        batch = {
            "feature": fbank.unsqueeze(0),
            "num_frames": torch.tensor([fbank.size(0)], dtype=torch.int64),
        }

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            embeddings, embedding_lens = model.get_embeddings(
                batch=batch,
                layer_idx=params.embedding_layer,
            )

        emb_np = embeddings[0, : int(embedding_lens[0])].detach().cpu().numpy()
        input_name = str(Path(wav_path).expanduser().resolve())
        save_name = f"{i:08d}_{_sanitize_filename(Path(wav_path).stem)}.npy"
        save_path = (params.output_dir / save_name).resolve()
        np.save(save_path, emb_np)
        mapping[input_name] = str(save_path)

        if (i + 1) % 50 == 0:
            logging.info(f"Processed {i + 1}/{len(wav_files)} wav files")

    logging.info(f"Finished extracting zipformer embeddings for {len(wav_files)} wav files")
    return mapping


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    params = AttributeDict()
    params.update(vars(args))
    params.output_dir = Path(params.output_dir)
    params.output_json = Path(params.output_json)
    params.wav_list = Path(params.wav_list)

    wav_files = load_wav_list(params.wav_list)
    print(f"Loaded {len(wav_files)} wav files")

    mapping = extract_embeddings(wav_files=wav_files, params=params)

    params.output_json.parent.mkdir(parents=True, exist_ok=True)
    with params.output_json.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Saved filename->npy mapping to {params.output_json}")
    
    