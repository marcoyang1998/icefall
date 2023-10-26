
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
from lhotse.features.io import NumpyHdf5Writer

import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
from typing import Union, Optional


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
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
        "--output-manifest",
        type=str,
        required=True,
        help="name of the manifest, e.g embeddings-dev-clean"
    )
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/embeddings"
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
    
    # whisper related args
    parser.add_argument(
        "--whisper-version",
        type=str,
        default="small.en"
    )
    
    return parser

def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log/log-whisper-embeddings")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.output_manifest}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.output_manifest}-{rank}.h5'
    else:
        output_manifest = params.embedding_dir / f"whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.output_manifest}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.output_manifest}.h5'
    
    device = torch.device("cuda", rank)
    
    # currently only use the encoder of whisper
    logging.info(params)
    model = whisper.load_model(params.whisper_version, device)
    model = model.encoder
    model.eval()
    logging.info(f"Number of whisper encoder params: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Successfully loaded Whisper model.")
    
    whisper_model = WhisperTeacher(model=model)
    
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
        logging.info(f"Writing Whisper embeddings to {embedding_path}")
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio = batch["audio"]
            audio_lens = batch["audio_lens"]
            
            embeddings, embedding_lens = whisper_model.get_embeddings(
                audio, 
                audio_lens,
                layer_idx=params.embedding_layer # which layer's embedding to be stored
            )
            embeddings = embeddings.detach().to("cpu").numpy()
            
            for idx, cut in enumerate(cuts):    
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.whisper_embedding = writer.store_array(
                    key=cut.id,
                    value=embeddings[idx][: embedding_lens[idx]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=0,
                )
                new_cuts.append(new_cut)
                num_cuts += 1
            if num_cuts and num_cuts % 100 == 0:
                logging.info(f"Cuts processed until now: {num_cuts}")
                
    logging.info(f"Finished extracting Whisper embeddings, processed a total of {num_cuts} cuts.")
                
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
        ori_cut.whisper_embedding = embed_cut.whisper_embedding
    
    input_cuts.to_jsonl(output_dir)
    print(f"Saved the joined manifest to {output_dir}")
    
def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 30.0:
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
    cuts = load_manifest(params.input_manifest)
    print(f"Finished loading manifest")
    
    target_manifest = params.embedding_dir / f"whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.output_manifest}.jsonl.gz"
    
    if not target_manifest.exists():
        if nj == 1:
            extract_embeddings(
                rank=0,
                manifest=cuts,
                params=params,    
            )
        else:
            splitted_cuts = cuts.split(num_splits=nj)
            print(f"Finished splitting manifest")
            mp.spawn(extract_embeddings, args=(splitted_cuts, params), nprocs=nj, join=True)
            manifests =  f"{str(params.embedding_dir)}/whisper-{params.whisper_version}-layer-{params.embedding_layer}-{params.output_manifest}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {target_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.input_manifest.replace(".jsonl.gz", f"-with-{params.whisper_version}-layer-{params.embedding_layer}-embeddings.jsonl.gz")
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=target_manifest,
            output_dir=output_manifest,
        )
    
    