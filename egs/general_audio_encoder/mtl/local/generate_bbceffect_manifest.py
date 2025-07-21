import argparse
import csv
import glob
import json
import logging
import os

import torch
from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment

from icefall.utils import str2bool

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="download/BBCSoundEffects",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/bbc_soundeffect_manifest",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"]
    )
    
    parser.add_argument(
        "--truncate",
        type=str2bool,
        help="If True, we truncate the audio into windows"
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="The window size for truncating"
    )

    return parser

def parse_meta(meta_file):
    # each meta file contains the sound description of the clip
    with open(meta_file, "r") as f:
        data = json.load(f)
    return data["text"]

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    split = args.split
    truncate = args.truncate
    window_size = args.window_size
    
    if split == "train":
        data_folder = f"{dataset_dir}/train/mnt/audio_clip/processed_datasets/BBCSoundEffects/train"
    else:
        data_folder = f"{dataset_dir}/test/mnt/audio_clip/processed_datasets/BBCSoundEffects/test"
    
    if truncate:
        manifest_output_dir = manifest_dir + "/" + f"bbc_soundeffect_cuts_{split}_{window_size}s.jsonl.gz"
    else:
        manifest_output_dir = manifest_dir + "/" + f"bbc_soundeffect_cuts_{split}.jsonl.gz"
        
    if os.path.exists(manifest_output_dir):
        logging.info(f"Maniest already created: {manifest_output_dir}! Please consider removing it.")
    else:
        audio_files = glob.glob(f"{data_folder}/*.flac")
        all_cuts = []
        num_cuts = 0
        for audio_file in audio_files:
            
            recording = Recording.from_file(audio_file)
            cut_id = split + "_" + audio_file.split("/")[-1].replace(".flac", "")
            cut = MonoCut(
                id=cut_id,
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
            )
            if truncate:
                # truncate the cut into windows
                cuts = cut.cut_into_windows(10)
                meta_file = audio_file.replace(".flac", ".json")
                text = parse_meta(meta_file)
                for window_cut in cuts:
                    sup = SupervisionSegment(
                        id=window_cut.id,
                        recording_id=cut.recording.id,
                        start=0.0, # always zero
                        channel=0,
                        duration=window_cut.duration,
                    )
                    sup.caption = text
                    window_cut.supervisions = [sup]
                    all_cuts.append(window_cut)
                    num_cuts += 1
                    if num_cuts % 100 == 0:
                        logging.info(f"Processed {num_cuts} cuts until now.")
            else:
                sup = SupervisionSegment(
                    id=cut_id,
                    recording_id=cut.recording.id,
                    start=0.0,
                    channel=0,
                    duration=cut.duration,
                )
                meta_file = audio_file.replace(".flac", ".json")
                text = parse_meta(meta_file)
                sup.caption = text
                cut.supervisions = [sup]
                all_cuts.append(cut)
                num_cuts += 1
            
                if num_cuts % 100 == 0:
                    logging.info(f"Processed {num_cuts} cuts until now.")
            
        logging.info(f"After filtering, a total of {len(all_cuts)} valid samples.")
        all_cuts = CutSet.from_cuts(all_cuts)
        all_cuts = all_cuts.resample(16000) # original dataset is 48k hz
        
        logging.info(f"Storing the manifest to {manifest_output_dir}")
        all_cuts.to_jsonl(manifest_output_dir)
    
    
    

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

