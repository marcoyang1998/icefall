import argparse
import csv
import glob
import logging
import os

import torch
from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording
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
        default="download/music4all",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/music4all_manifest",
    )
    
    return parser

def parse_meta_info(dataset_dir: str):
    info_list = ["genres", "lang", "information", "tags"]
    merged_df = None
    for info in info_list:
        csv_file = f"{dataset_dir}/id_{info}.csv"
        df = pd.read_csv(csv_file, sep='\t')
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="id", how="inner")
    return merged_df


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    
    os.makedirs(manifest_dir, exist_ok=True)
    
    meta_info = parse_meta_info(dataset_dir)
    
    cuts = []
    num_cuts = 0
    for _, row in meta_info.iterrows():
        id = row["id"]
        
        audio_file = f"{dataset_dir}/audios/{id}.mp3"
        recording = Recording.from_file(audio_file)
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
            language=row["lang"],
        )
        supervision.genres = row["genres"]
        supervision.artist = row["artist"]
        supervision.song = row["song"]
        supervision.album = row["album_name"]
        supervision.tags = row["tags"]
        cut.supervisions = [supervision]
        
        cuts.append(cut)
        num_cuts += 1
        if num_cuts % 100 == 0:
            logging.info(f"Processed {num_cuts} cuts until now.")
        
    logging.info(f"After filtering, a total of {len(cuts)} valid samples.")
    cuts = CutSet.from_cuts(cuts)
    cuts = cuts.resample(16000)
    
    manifest_output_dir = manifest_dir + "/" + f"music4all_cuts_all.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
