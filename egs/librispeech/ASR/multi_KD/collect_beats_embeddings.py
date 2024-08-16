
import argparse
import os
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger, make_pad_mask

import torch
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import DataLoader

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler, UnsupervisedWaveformDataset

from BEATs import BEATs, BEATsConfig

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=1000,
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
        help="name of the manifest, e.g embeddings-balanced"
    )
    
    parser.add_argument(
        "--beats-ckpt",
        type=str,
        default="data/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    )
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/embeddings"
    )

    parser.add_argument(
        "--target-manifest-file",
        type=str,
        required=True,
        help="Where to store the manifest augmented with whisper features"
    )
    
    return parser
    
    
def get_embeddings():
    checkpoint = torch.load('data/models/BEATs_iter3_plus_AS2M.pt')

    cfg = BEATsConfig(checkpoint['cfg'])
    
    device = torch.device("cuda")
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    BEATs_model.to(device)

    # extract the the audio representation
    audio_input_16khz, _ = torchaudio.load('/star-xy/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac')
    audio_input_16khz = audio_input_16khz.to(device)
    padding_mask = torch.zeros_like(audio_input_16khz).bool().to(device)
    
    representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
    print(representation.shape)

@torch.no_grad()
def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log/log-beats-embeddings")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"{params.model_id}-{params.manifest_name}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'{params.model_id}-{params.manifest_name}-{rank}.h5'
    else:
        output_manifest = params.embedding_dir / f"{params.model_id}-{params.manifest_name}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'{params.model_id}-{params.manifest_name}.h5'
    
    logging.info(params)
    
    checkpoint = torch.load(params.beats_ckpt)
    cfg = BEATsConfig(checkpoint['cfg'])
    logging.info(f"Successfully load BEATs model.")
    
    device = torch.device("cuda", rank)
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    BEATs_model.to(device)    
    
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
        logging.info(f"Writing BEATs embeddings to {embedding_path}")
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio_input_16khz = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            padding_mask = make_pad_mask(audio_lens)
            
            embeddings = BEATs_model.extract_features(
                audio_input_16khz, padding_mask=padding_mask
            )[0].detach().to("cpu").numpy() # (N, C)
            
            for idx, cut in enumerate(cuts):
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.beats_embedding = writer.store_array(
                    key=cut.id,
                    value=embeddings[idx],
                )
                new_cuts.append(new_cut)
                num_cuts += 1
            if num_cuts and num_cuts % 100 == 0:
                logging.info(f"Cuts processed until now: {num_cuts}")
    logging.info(f"Finished extracting BEATs embeddings, processed a total of {num_cuts} cuts.")
                
    CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)
    logging.info(f"Saved manifest to {output_manifest}")
    
def join_manifests(
    input_cuts: CutSet,
    embedding_manifest: str,
    output_dir: str,
):
    embedding_cuts = load_manifest(embedding_manifest)
    
    assert len(embedding_cuts) == len(input_cuts)
    assert set(input_cuts.ids) == set(embedding_cuts.ids)
    
    embedding_cuts = embedding_cuts.sort_like(input_cuts)
    for cut_idx, (ori_cut, embed_cut) in enumerate(zip(input_cuts, embedding_cuts)):
        assert ori_cut.id == embed_cut.id
        ori_cut.beats_embedding = embed_cut.beats_embedding
    
    input_cuts.to_jsonl(output_dir)
    print(f"Saved the joined manifest to {output_dir}")


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
    
    params.model_id = params.beats_ckpt.split("/")[-1].replace(".pt", "")
    embedding_manifest = params.embedding_dir / f"{params.model_id}-{params.manifest_name}.jsonl.gz"
    
    if not embedding_manifest.exists():
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
            manifests =  f"{str(params.embedding_dir)}/{params.model_id}-{params.manifest_name}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {embedding_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.target_manifest_file
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=embedding_manifest,
            output_dir=output_manifest,
        )
    else:
        print("The output manifest file already exists!")
    
    
    
        
    
        
        
        
        