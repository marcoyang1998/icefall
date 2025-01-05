
import argparse
import os
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger
from teachers import WhisperTeacher

import torch
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import DataLoader

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.dataset import SimpleCutSampler, UnsupervisedWaveformDataset, DynamicBucketingSampler
from lhotse.features.io import LilcomChunkyWriter, NumpyHdf5Writer
from lhotse.utils import fastcopy
import multi_quantization as quantization
import numpy as np

import whisper
from typing import Union, Optional


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # quantizer related
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
    )
    
    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=4,
    )
    
    parser.add_argument(
        "--quantizer-path",
        type=str,
        required=True,
    )
    
    # others
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--manifest-name",
        type=str,
        required=True,
        help="name of the manifest, e.g embeddings-dev-clean, embeddings-train-clean-100"
    )
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/vq_whisper"
    )

    parser.add_argument(
        "--embedding-layer",
        type=int,
        default=-1,
        help="Which layer's representation should be extracted",
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--target-manifest-file",
        type=str,
        required=True,
        help="Where to store the manifest augmented with whisper features"
    )
    
    # whisper related args
    parser.add_argument(
        "--whisper-version",
        type=str,
        default="turbo"
    )

    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
    )
    
    return parser

@torch.no_grad()
def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/vq_whisper/log/log-whisper-cb-indexes")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}'
    else:
        output_manifest = params.embedding_dir / f"whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.manifest_name}'
    
    device = torch.device("cuda", rank)
    
    # currently only use the encoder of whisper
    logging.info(params)
    model = whisper.load_model(params.whisper_version, device)
    model = model.encoder
    model.eval()
    logging.info(f"Number of whisper encoder params: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Successfully loaded Whisper model.")
    
    whisper_model = WhisperTeacher(model=model, n_mels=params.n_mels)
    
    quantizer = quantization.Quantizer(
        dim=params.embedding_dim,
        num_codebooks=params.num_codebooks,
        codebook_size=256,
    )
    quantizer.load_state_dict(torch.load(params.quantizer_path))
    quantizer.to(device)
    
    dataset = UnsupervisedWaveformDataset(
        manifest
    )
    
    sampler = DynamicBucketingSampler(
        manifest,
        max_duration=params.max_duration,
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
    
    new_cuts = []
    num_cuts = 0
    
    with NumpyHdf5Writer(embedding_path) as writer:
        logging.info(f"Writing Whisper indexes to {embedding_path}")
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio = batch["audio"]
            audio_lens = batch["audio_lens"]
            
            embeddings, embedding_lens = whisper_model.get_embeddings(
                audio, 
                audio_lens,
                layer_idx=params.embedding_layer # which layer's embedding to be stored
            )
            
            # codebook_indexes = quantizer.encode(embeddings) # [N, T, C]
            N,T,C = embeddings.shape
            embeddings = embeddings.reshape(-1, C)
            B = 2000
            splits = embeddings.split(B)
            codebook_indexes = []
            for chunk in splits:
                chunk_indexes = quantizer.encode(chunk)
                codebook_indexes.append(chunk_indexes)
            codebook_indexes = torch.cat(codebook_indexes).reshape(N,T,params.num_codebooks)
            codebook_indexes = codebook_indexes.to("cpu").numpy()
            assert np.min(codebook_indexes) >= 0
            assert np.max(codebook_indexes) < 256
            
            for idx, cut in enumerate(cuts):    
                cb_index = writer.store_array(
                    key=cut.id,
                    value=codebook_indexes[idx][: embedding_lens[idx]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=cut.start,
                )
                new_cut = fastcopy(
                    cut,
                    custom={"codebook_indexes": cb_index}
                )
                new_cuts.append(new_cut)
                num_cuts += 1
                if num_cuts and num_cuts % 100 == 0:
                    logging.info(f"Cuts processed until now: {num_cuts}")
                    # torch.cuda.empty_cache()
                
    logging.info(f"Finished extracting Whisper codebook indexes, processed a total of {num_cuts} cuts.")
                
    CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)
    logging.info(f"Saved manifest to {output_manifest}")
    
def join_manifests(
    input_cuts: CutSet,
    embedding_manifest: str,
    output_dir: str,
):
    # Combine the teacher embedding manifest with the original manifest for ASR
    embedding_cuts = load_manifest(embedding_manifest)
    
    assert len(embedding_cuts) == len(input_cuts)
    assert set(input_cuts.ids) == set(embedding_cuts.ids)
    
    embedding_cuts = embedding_cuts.sort_like(input_cuts)
    for cut_idx, (ori_cut, embed_cut) in enumerate(zip(input_cuts, embedding_cuts)):
        assert ori_cut.id == embed_cut.id
        ori_cut.codebook_indexes = embed_cut.codebook_indexes
    
    input_cuts.to_jsonl(output_dir)
    logging.info(f"Saved the joined manifest to {output_dir}")
    
def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 24.0:
        return False
    return True

def remove_sp(c):
    if "sp0.9" in c.id or "sp1.1" in c.id:
        return False
    return True

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    params.embedding_dir = Path(params.embedding_dir)
    
    nj = params.num_jobs
    print(f"Start loading manifest")
    cuts = load_manifest(params.input_manifest)
    cuts = cuts.filter(remove_short_and_long_utt) # remove audio longer than 30s
    cuts = cuts.filter(remove_sp) # remove speed perturb
    print(f"Finished loading manifest")
    print(cuts)
    
    embedding_manifest = params.embedding_dir / f"whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
    
    if not embedding_manifest.exists():
        if nj == 1:
            extract_embeddings(
                rank=0,
                manifest=cuts,
                params=params,    
            )
        else:
            splitted_cuts = cuts.split(num_splits=nj)
            logging.info(f"Finished splitting manifest")
            mp.spawn(extract_embeddings, args=(splitted_cuts, params), nprocs=nj, join=True)
            manifests =  f"{str(params.embedding_dir)}/whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.manifest_name}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {embedding_manifest}")
    else:
        logging.info(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.target_manifest_file
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=embedding_manifest,
            output_dir=output_manifest,
        )
    
    