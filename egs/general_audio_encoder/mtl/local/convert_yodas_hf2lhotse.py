import argparse
import glob
import logging
import os
from tqdm import tqdm
import subprocess

from datasets import load_dataset
from lhotse import CutSet
import torch

from contextlib import contextmanager

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        default="en"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=8,
    )
    return parser.parse_args()

@contextmanager
def get_executor():
    # We'll either return a process pool or a distributed worker pool.
    # Note that this has to be a context manager because we might use multiple
    # context manager ("with" clauses) inside, and this way everything will
    # free up the resources at the right time.
    try:
        # If this is executed on the CLSP grid, we will try to use the
        # Grid Engine to distribute the tasks.
        # Other clusters can also benefit from that, provided a
        # cluster-specific wrapper.
        # (see https://github.com/pzelasko/plz for reference)
        #
        # The following must be installed:
        # $ pip install dask distributed
        # $ pip install git+https://github.com/pzelasko/plz
        name = subprocess.check_output("hostname -f", shell=True, text=True)
        if name.strip().endswith(".clsp.jhu.edu"):
            import plz
            from distributed import Client

            with plz.setup_cluster() as cluster:
                cluster.scale(80)
                yield Client(cluster)
            return
    except Exception:
        pass
    # No need to return anything - compute_and_store_features
    # will just instantiate the pool itself.
    yield None

def trim_yodas_dataset(
    lang: str = "en", 
    start: int = -1,
    end: int = -1,
    num_jobs: int = 8,
):
    assert start <= end
    # num_jobs = min(8, os.cpu_count())
    # num_jobs = 8
    logging.info(f"Using {num_jobs} jobs.")
    
    def process_cut(c):
        c.id = c.utt_id
        return c
    
    parquet_files = []
    for split in range(start, end + 1):
        files = glob.glob(f"download/yodas-granary/data/{lang}{str(split).zfill(3)}/asr_only/*.parquet")
        parquet_files += files
    
    parquet_files = sorted(parquet_files)
    logging.info(f"Found a total of {len(parquet_files)} splits.")
    
    os.makedirs(f"data/yodas_granary/{lang}", exist_ok=True)

    for i, cur_parquet in enumerate(tqdm(parquet_files)):
        # get the split and parquet id
        split = cur_parquet.split("/")[-3]
        parquet_id = cur_parquet.split("/")[-1].split(".")[0]
        
        os.makedirs(f"data/yodas_granary/{lang}/{split}", exist_ok=True)
        output_manifest_name = f"data/yodas_granary/{lang}/{split}/yodas_granary_cuts_train_{split}_{parquet_id}.jsonl.gz"
        if os.path.exists(output_manifest_name):
            logging.info(f"Manifest already created, skipping {output_manifest_name}")
            continue
        
        logging.info(f"Start loading: {cur_parquet}")
        try:
            ds = load_dataset(
                "parquet",
                data_files=cur_parquet,
                # streaming=True,
                split="train",
            )
            logging.info(f"Finish loading: {cur_parquet}")
            cuts = CutSet().from_huggingface_dataset(
                ds,
                audio_key="audio",
                text_key="text",
                lang_key="lang"
            )
        except:
            logging.info(f"Error when loading the {cur_parquet}, skip it for now")
            continue
        
        # change some attributes
        cuts = cuts.map(process_cut)
        
        with get_executor() as ex:
            cur_nj = min(len(ds), num_jobs)
            cuts = cuts.save_audios(
                format="wav",
                storage_path=f"download/yodas-granary-trimmed/{split}/{parquet_id}",
                num_jobs=cur_nj if ex is None else 80,
                executor=ex,
                progress_bar=False
            )
        cuts.to_jsonl(output_manifest_name)
        logging.info(f"Saved to {output_manifest_name}")
        

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    lang = args.lang
    start = args.start
    end = args.end
    num_jobs = args.num_jobs
    
    trim_yodas_dataset(lang, start=start, end=end, num_jobs=num_jobs)