import logging
import argparse

import torch
from WavLM import WavLM, WavLMConfig

from lhotse import load_manifest_lazy, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer, LilcomChunkyWriter
from lhotse.utils import fastcopy
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from utils import make_pad_mask

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--wavlm-version",
        type=str,
        required=True,
        choices=["base", "base-plus", "large"]
    )
    
    parser.add_argument(
        "--wavlm-ckpt",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="Index starts from 1, to get the 12-th layer features, set layer-idx=12"
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
    )
    
    return parser.parse_args()

def test_wavlm():
    # load the pre-trained checkpoints
    checkpoint = torch.load('WavLM-Base+.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    device = torch.device("cuda")
    model.to(device)

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1,10000).to(device)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
    rep = model.extract_features(wav_input_16khz)[0]

    # extract the representation of each layer
    wav_input_16khz = torch.randn(1,10000)
    padding_mask = torch.zeros(1, 10000).bool()
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
    rep, layer_results = model.extract_features(wav_input_16khz, padding_mask=padding_mask, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
    layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
    
@torch.no_grad()
def collect_results(ckpt_path, manifest_path, embedding_path, output_manifest_path, layer_idx=21, max_duration=200):
    
    checkpoint = torch.load(ckpt_path)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    manifest = load_manifest_lazy(manifest_path)
    dataset = UnsupervisedWaveformDataset(
        manifest
    )
    
    sampler = DynamicBucketingSampler(
        manifest,
        max_duration=max_duration,
        shuffle=False,
        drop_last=False,
    )
    
    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=1,
        persistent_workers=False,
    )
    
    device = torch.device("cuda")
    model.to(device)
    
    new_cuts = []
    num_cuts = 0
    with LilcomChunkyWriter(embedding_path) as writer:
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio_input_16khz = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            padding_mask = make_pad_mask(audio_lens)
            
            if cfg.normalize:
                audio_input_16khz = torch.nn.functional.layer_norm(audio_input_16khz, audio_input_16khz.shape)
            
            (rep, layer_results), padding_mask = model.extract_features(
                audio_input_16khz,
                padding_mask=padding_mask,
                output_layer=model.cfg.encoder_layers,
                ret_layer_results=True
            )
            
            layer_results = [res.permute(1,0,2).cpu().numpy() for res, _ in layer_results] # list of (B,T,C)
            layer_results = layer_results[layer_idx] # (B,T,C)
            embedding_lens = (~padding_mask).sum(dim=-1)
            
            for j, cut in enumerate(cuts):
                wavlm_embedding = writer.store_array(
                    key=cut.id,
                    value=layer_results[j][:embedding_lens[j]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=0,
                )
                new_cut = fastcopy(
                    cut,
                    custom={"wavlm_embedding": wavlm_embedding}
                )
                new_cuts.append(new_cut)
                num_cuts += 1
                if num_cuts and num_cuts % 200 == 0:
                    logging.info(f"Cuts processed until now: {num_cuts}")
    
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest_path)
    logging.info(f"Manifest saved to {output_manifest_path}")
            
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    wavlm_version = args.wavlm_version
    ckpt_path = args.wavlm_ckpt
    layer_idx = args.layer_idx
    subset = args.subset
    
    manifest_path = f"data/fbank/librispeech_cuts_{subset}.jsonl.gz"
    embedding_path = f"embeddings/wavlm_embeddings/wavlm-{wavlm_version}-layer-{layer_idx}-{subset}.h5"
    output_manifest_path = f"manifests/{subset}-wavlm-{wavlm_version}-layer-{layer_idx}.jsonl.gz"
    
    max_duration = 200
    collect_results(
        ckpt_path=ckpt_path,
        manifest_path=manifest_path,
        embedding_path=embedding_path,
        output_manifest_path=output_manifest_path,
        layer_idx=layer_idx,
        max_duration=max_duration,
    )