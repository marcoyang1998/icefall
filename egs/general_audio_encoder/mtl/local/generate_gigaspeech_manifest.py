import argparse
import csv
import glob
import logging
import os
from tqdm import tqdm

import pandas as pd
import torch
from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.audio.utils import AudioLoadingError
from lhotse.supervision import SupervisionSegment
import pandas as pd

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="download/gigaspeech",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/gigaspeech_manifest",
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="dev"
    )
    
    return parser

def parse_metadata(meta_file: str):
    df = pd.read_csv(meta_file, sep=",")
    df = df[["sid", "text_tn", "begin_time", "end_time"]]
    return df
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    subset = args.subset
    
    os.makedirs(manifest_dir, exist_ok=True)
    
    
    if subset in ["dev", "test", "xs"]:
        meta_root_dir = f"{dataset_dir}/data/metadata/{subset}_metadata"
        audio_root_dir = f"{dataset_dir}/data/audio/{subset}_files"
    else:
        meta_root_dir = f"{dataset_dir}/data/metadata/{subset}_metadata_additional"
        audio_root_dir = f"{dataset_dir}/data/audio/{subset}_files_additional"
    
    meta_files = sorted(glob.glob(f"{meta_root_dir}/*.csv"))
    
    all_cuts = []
    num_cuts = 0
    for meta_file in tqdm(meta_files):
        chunk_folder = meta_file.replace("_metadata.csv", "")
        meta_info = parse_metadata(meta_file)
        for _, row in meta_info.iterrows():
            audio_id = row["sid"]
            audio_file = f"{audio_root_dir}/{chunk_folder}/{audio_id}.wav"
            import pdb; pdb.set_trace()
        
            try:
                recording = Recording.from_file(audio_file)
            except AudioLoadingError:
                logging.info(f"Error when loading {audio_file}. Skip it")
                continue
                
            cut = MonoCut(
                id=id,
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
            )
            supervision = SupervisionSegment(
                id=id,
                recording_id=cut.recording.id,
                start=0.0,
                channel=0,
                duration=cut.duration,
                text=row["transcript"],
                language="en",
            )
            cut.supervisions = [supervision]
            
            all_cuts.append(cut)
            num_cuts += 1
            if num_cuts % 100 == 0:
                logging.info(f"Processed {num_cuts} cuts until now.")
        
    logging.info(f"After filtering, a total of {len(all_cuts)} valid samples.")
    cuts = CutSet.from_cuts(all_cuts)
    manifest_output_dir = manifest_dir + "/" + f"gigaspeech_cuts_{subset}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()