import argparse
import csv
import glob
import logging
import os

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
        default="download/vggsound_10s",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/vggsound_manifest",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"]
    )
    
    parser.add_argument(
        "--meta-file",
        type=str,
        default="download/vggsound.csv"
    )
    
    return parser

def parse_meta_file(meta_file: str, split: str):
    df = pd.read_csv(meta_file, comment='#', header=None,
                 names=['ytid', 'start', 'label', 'split'])
    df = df[df["split"] == split]
    return df

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    meta_file = args.meta_file
    split = args.split
    
    os.makedirs(manifest_dir, exist_ok=True)
    
    logging.info(f"Processing {split}")
    data = parse_meta_file(meta_file, split)
    logging.info(f"Find a total of {len(data)} clips.")
    
    cuts = []
    num_cuts = 0
    for _, row in data.iterrows():
        ytid = row["ytid"]
        start = row["start"]
        label = row["label"]
        end = start + 10
        
        audio_filename = f"{dataset_dir}/{ytid}_{start:.2f}_{end:.2f}.wav"
        if not os.path.exists(audio_filename):
            logging.info(f"{audio_filename} does not exist. Skip it!")
            continue
        
        try:
            recording = Recording.from_file(audio_filename)
        except AudioLoadingError:
            logging.info(f"Audiofile: {audio_filename} broken!")
            continue
        cut_id = f"{ytid}_{start}_{end}"
        cut = MonoCut(
            id=cut_id,
            start=0,
            duration=recording.duration,
            channel=0,
            recording=recording,
        )
        supervision = SupervisionSegment(
            id=cut_id,
            recording_id=cut.recording.id,
            start=0.0,
            channel=0,
            duration=cut.duration,
        )
        supervision.audio_event = label
        cut.supervisions = [supervision]
        cuts.append(cut)
        num_cuts += 1
        
        if num_cuts % 100 == 0:
            logging.info(f"Processed {num_cuts} cuts until now.")
            
    logging.info(f"After filtering, a total of {len(cuts)}/{len(data)} valid samples.")
    cuts = CutSet.from_cuts(cuts)
    
    manifest_output_dir = manifest_dir + "/" + f"vggsound_cuts_{split}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
        
        
        
    
    
    
    