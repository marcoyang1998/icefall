import argparse
import os
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger
from teachers import MertTeacher

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import DataLoader
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.dataset import SimpleCutSampler, UnsupervisedWaveformDataset, DynamicBucketingSampler
from lhotse.features.io import NumpyHdf5Writer, LilcomChunkyWriter

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
        help="The input jsonl.gz file"
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
        "--embedding-level",
        type=str,
        default="clip",
        choices=["clip", "frame"],
        help="Use the which level's embedding",
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=500,
    )
    
    # MERT related args
    parser.add_argument(
        "--mert-version",
        type=str,
        default="MERT-v1-330M",
        choices=["MERT-v1-330M", "MERT-v1-95M"]
    )
    
    return parser

def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log/log-mert-embeddings")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"mert-{params.mert_version}-layer-{params.embedding_layer}-{params.output_manifest}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'mert-{params.mert_version}-layer-{params.embedding_layer}-{params.output_manifest}-{rank}.h5'
    else:
        output_manifest = params.embedding_dir / f"mert-{params.mert_version}-layer-{params.embedding_layer}-{params.output_manifest}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'mert-{params.mert_version}-layer-{params.embedding_layer}-{params.output_manifest}.h5'
    
    device = torch.device("cuda", rank)
    
    logging.info(params)

    # load the MERT model
    model = AutoModel.from_pretrained(f"m-a-p/{params.mert_version}", trust_remote_code=True)
    model.eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(f"m-a-p/{params.mert_version}",trust_remote_code=True)

    sampling_rate = manifest[0].sampling_rate
    if sampling_rate != processor.sampling_rate:
        manifest = manifest.resample(processor.sampling_rate)
        logging.info(f"Resample the audio from {sampling_rate} to {processor.sampling_rate}.")

    logging.info(f"Successfully loaded MERT model.")
    mert_model = MertTeacher(model=model, processor=processor)
    mert_model.to(device)
    
    dataset = UnsupervisedWaveformDataset(
        collate=False,
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
        logging.info(f"Writing MERT embeddings to {embedding_path}")
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio = batch["audio"]
            audio = [a.squeeze().numpy() for a in audio]
            audio_lens = np.array([a.shape[0] for a in audio])
            
            embeddings = mert_model.get_embeddings(
                audio, 
                sampling_rate=cuts[0].sampling_rate,
                level=params.embedding_level,
                layer_idx=params.embedding_layer # which layer's embedding to be stored
            )
            embeddings = embeddings.detach().to("cpu").numpy()

            embedding_lens = audio_lens // 320 # this is a rough estimation
            
            for idx, cut in enumerate(cuts):    
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    recording=cut.recording,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.mert_embedding = writer.store_array(
                    key=cut.id,
                    value=embeddings[idx][: embedding_lens[idx]],
                    temporal_dim=0,
                    frame_shift=1/75, # the output frame rate is 75hz
                    start=cut.start,
                )
                new_cuts.append(new_cut)
                num_cuts += 1
            if num_cuts and num_cuts % 100 == 0:
                logging.info(f"Cuts processed until now: {num_cuts}")
                
    logging.info(f"Finished extracting MERT embeddings, processed a total of {num_cuts} cuts.")
                
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
    if c.duration < 1.0 or c.duration > 31.0:
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
    
    target_manifest = params.embedding_dir / f"MERT-{params.mert_version}-layer-{params.embedding_layer}-{params.output_manifest}.jsonl.gz"
    
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
            manifests =  f"{str(params.embedding_dir)}/MERT-{params.mert_version}-layer-{params.embedding_layer}-{params.output_manifest}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {target_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.input_manifest.replace(".jsonl.gz", f"-with-MERT-{params.mert_version}-layer-{params.embedding_layer}-embeddings.jsonl.gz")
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=target_manifest,
            output_dir=output_manifest,
        )
    
    