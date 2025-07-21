import argparse
import os
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List

import torch
import torch.multiprocessing as mp
from lhotse import load_manifest_lazy, load_manifest, CutSet
from lhotse.features.io import NumpyHdf5Writer
from lhotse.utils import fastcopy
import numpy as np

from icefall.utils import AttributeDict

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
        "--manifest-dir",
        type=str,
        default="data/vq_whisper"
    )
    
    parser.add_argument(
        "--target-manifest-file",
        type=str,
        required=True,
        help="Where to store the manifest augmented with whisper features"
    )
    return parser

def load_codebook_indexes(c):
    info = c.codebook_indexes
    if isinstance(info, dict):
        filename = info["path"]
        cb_indexes = np.load(filename)
    else:
        cb_indexes = c.load_custom("codebook_indexes")

    return cb_indexes


def convert_npy_to_hdf5(
    rank: int,
    manifest: List[CutSet],
    params: AttributeDict,
):
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.manifest_dir / f"{params.manifest_name}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f"{params.manifest_name}-{rank}"
    else:
        output_manifest = params.manifest_dir / f"{params.manifest_name}.jsonl.gz"
        embedding_path = params.embedding_dir / params.manifest_name
    
    new_cuts = []
    num_cuts = 0
    with NumpyHdf5Writer(embedding_path) as writer:
        for cut in manifest:
            codebook_indexes = load_codebook_indexes(cut)
            assert np.min(codebook_indexes) >= 0
            assert np.max(codebook_indexes) < 256
            assert codebook_indexes.dtype == np.uint8
            cb_index = writer.store_array(
                key=cut.id,
                value=codebook_indexes,
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
                print(f"Rank {rank}: cuts processed until now: {num_cuts}")
    
    print(f"Rank {rank}: finished converting npy to hdf5, processed a total of {num_cuts} cuts.")
                
    CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)
    print(f"Rank {rank}: saved manifest to {output_manifest}")

    
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    params.manifest_dir = Path(params.manifest_dir)
    params.embedding_dir =  params.manifest_dir / params.manifest_name
    params.embedding_dir.mkdir(parents=True, exist_ok=True)
    
    nj = params.num_jobs
    print(f"Start loading manifest: {params.input_manifest}")
    cuts = load_manifest(params.input_manifest)
    print(f"Finished loading manifest")
    print(cuts)
    
    output_manifest = Path(params.target_manifest_file)
    
    # if not output_manifest.exists():
    #     if nj == 1:
    #         convert_npy_to_hdf5(
    #             rank=0,
    #             manifest=cuts,
    #             params=params,    
    #         )
    #     else:
    #         splitted_cuts = cuts.split(num_splits=nj)
    #         logging.info(f"Finished splitting manifest")
    #         mp.spawn(convert_npy_to_hdf5, args=(splitted_cuts, params), nprocs=nj, join=True)
    #         import pdb; pdb.set_trace()
    #         manifests =  f"{str(params.manifest_dir)}/{params.manifest_name}-*.jsonl.gz"
    #         os.system(f"lhotse combine {manifests} {str(output_manifest)}")
    #         logging.info(f"Saved to {str(output_manifest)}")
    # else:
    #     logging.info(f"Skip embedding extraction: the manifest is already generated.")
    
    if not output_manifest.exists():
        if nj == 1:
            convert_npy_to_hdf5(
                rank=0,
                manifest=cuts,
                params=params,
            )
        else:
            splitted_cuts = cuts.split(num_splits=nj)
            logging.info("Finished splitting manifest")

            processes = []
            for rank in range(nj):
                p = mp.Process(
                    target=convert_npy_to_hdf5,
                    args=(rank, splitted_cuts, params)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # 合并 jsonl.gz 文件
            manifests = f"{str(params.manifest_dir)}/{params.manifest_name}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {str(output_manifest)}")
            logging.info(f"Saved to {str(output_manifest)}")
    else:
        logging.info("Skip embedding extraction: the manifest is already generated.")
    
    