
import argparse
import os
import logging
from pathlib import Path

from kd_datamodule import LibriSpeechKDDataModule
from icefall.utils import AttributeDict, setup_logger, make_pad_mask

import torch
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import DataLoader

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler, UnsupervisedWaveformDataset

import models
from models.checkpoints import list_models

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
        "--ced-ckpt",
        type=str,
        default="pretrained_models/CED/averaged.pt",
    )
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/embeddings"
    )
    
    return parser
    

@torch.no_grad()
def extract_embeddings(
    rank: int,
    data_module,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"{params.embedding_dir}/log/log-ced-embeddings")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"{params.model_id}-{params.output_manifest}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'{params.model_id}-{params.output_manifest}-{rank}.h5'
    else:
        output_manifest = params.embedding_dir / f"{params.model_id}-{params.output_manifest}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'{params.model_id}-{params.output_manifest}.h5'
    
    device = torch.device("cuda", rank)

    logging.info(params)

    model = getattr(models, "ced_base")(
        pretrained=True,
        pretrained_url=params.ced_ckpt,
        n_mels=80,
    ).to(device)
    model.eval()
    
    # dataset = UnsupervisedWaveformDataset(
    #     manifest
    # )
    
    # sampler = DynamicBucketingSampler(
    #     manifest,
    #     max_duration=params.max_duration,
    #     shuffle=False,
    #     drop_last=False,
    # )
    
    # dl = DataLoader(
    #     dataset,
    #     sampler=sampler,
    #     batch_size=None,
    #     num_workers=1,
    #     persistent_workers=False,
    # )
    dl = data_module.valid_dataloaders(manifest)
    
    new_cuts = []
    num_cuts = 0
    
    with NumpyHdf5Writer(embedding_path) as writer:
        logging.info(f"Writing CED embeddings to {embedding_path}")
        for i, batch in enumerate(dl):
            cuts = batch["supervisions"]["cut"]
            feature = batch["inputs"].to(device)
            feature = feature.permute(0,2,1)
            #audio_input_16khz = batch["audio"].to(device)
            #audio_lens = batch["audio_lens"].to(device)
            #padding_mask = make_pad_mask(audio_lens)
            
            embeddings = model.forward_spectrogram(feature).detach().to("cpu").numpy() # (N, C)
            
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
    logging.info(f"Finished extracting CED embeddings, processed a total of {num_cuts} cuts.")
                
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
    LibriSpeechKDDataModule.add_arguments(parser)
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    params.embedding_dir = Path(params.embedding_dir)
    
    print("Start loading manifest")
    nj = params.num_jobs
    cuts = load_manifest(params.input_manifest)
    print(f"Finished loading manifest")
    
    params.model_id = "CED-base"
    target_manifest = params.embedding_dir / f"{params.model_id}-{params.output_manifest}.jsonl.gz"

    args.return_cuts = True
    librispeech = LibriSpeechKDDataModule(args, evaluation=True)
    
    if not target_manifest.exists():
        if nj == 1:
            extract_embeddings(
                data_module=librispeech,
                rank=0,
                manifest=cuts,
                params=params,    
            )
        else:
            splitted_cuts = cuts.split(num_splits=nj)
            print(f"Finished splitting manifest")
            mp.spawn(extract_embeddings, args=(librispeech, splitted_cuts, params), nprocs=nj, join=True)
            manifests =  f"{str(params.embedding_dir)}/{params.model_id}-{params.output_manifest}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {target_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.input_manifest.replace(".jsonl.gz", "-with-CED-embeddings.jsonl.gz")
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=target_manifest,
            output_dir=output_manifest,
        )
    
    
    
        
    
        
        
        
        