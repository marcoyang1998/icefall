
import argparse
import os
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger
from model import MyBertModel

import h5py
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.dataset import UnsupervisedWaveformDataset, DynamicBucketingSampler


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
        "--manifest-name",
        type=str,
        required=True,
        help="name of the manifest, e.g embeddings-dev-clean, embeddings-train-clean-100"
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
        help="Which layer's representation should be extracted, index start from 1, i.e the 10-th layer requires"
        "--embedding-layer 10"
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
        "--bert-version",
        type=str,
        default="large"
    )
    
    return parser

@torch.no_grad()
def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log/log-bert-embeddings")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"bert-{params.bert_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'bert-{params.bert_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}.h5'
    else:
        output_manifest = params.embedding_dir / f"bert-{params.bert_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'bert-{params.bert_version}-layer-{params.embedding_layer}-{params.manifest_name}.h5'
    
    device = torch.device("cuda", rank)
    
    # currently only use the encoder of whisper
    logging.info(params)
    model = MyBertModel(model_version=params.bert_version)
    feature_dim = model.model_dim
    model.to(device)
    model.eval()
    logging.info(f"Number of bert model params: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Successfully loaded bert model.")
    
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
        num_workers=2,
        persistent_workers=False,
    )
    
    new_cuts = []
    num_cuts = 0
    
    # with NumpyHdf5Writer(embedding_path) as writer:
    with h5py.File(embedding_path, "w") as writer:
        # dt = h5py.vlen_dtype(np.dtype(f"float32,({feature_dim},)"))  # 每行是 C 维向量，长度变
        
        grp = writer.create_group("embeddings")
        
        logging.info(f"Writing bert embeddings to {embedding_path}")
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            texts = [c.supervisions[0].text for c in cuts]
            
            embeddings, embedding_lens = model.get_embeddings(
                texts=texts,
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
                grp.create_dataset(
                    cut.id,
                    data=embeddings[idx][: embedding_lens[idx]],
                    compression="lzf"
                )
                embed_meta_info = {
                    "key": cut.id,
                    "file": str(embedding_path),
                    "type": "hdf5",
                }
                new_cut.embedding = embed_meta_info
                new_cuts.append(new_cut)
                num_cuts += 1
            if num_cuts and i % 100 == 0:
                logging.info(f"Cuts processed until now: {num_cuts}")
                
    logging.info(f"Finished extracting bert embeddings, processed a total of {num_cuts} cuts.")
                
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
        ori_cut.embedding = embed_cut.embedding
    
    input_cuts.to_jsonl(output_dir)
    print(f"Saved the joined manifest to {output_dir}")
    
def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 29.9:
        return False
    return True

def remove_sp(c):
    if "sp1.1" in c.id or "sp0.9" in c.id:
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
    cuts = cuts.filter(remove_short_and_long_utt) # remove audio longer than 30s
    cuts = cuts.filter(remove_sp) # remove the speed perturbed audio
    print(f"Finished loading manifest")
    
    embedding_manifest = params.embedding_dir / f"bert-{params.bert_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
    
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
            manifests =  f"{str(params.embedding_dir)}/bert-{params.bert_version}-layer-{params.embedding_layer}-{params.manifest_name}-*.jsonl.gz"
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
    
    