import argparse
import csv
import glob
import logging
import os

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
        default="download/spgispeech",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/spgispeech_manifest",
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="dev"
    )
    
    return parser

def generate_audio_mapping(audio_root: str):
    audio_files = glob.glob(f"{audio_root}/*/*/*.wav")
    def get_filename(s: str):
        return "/".join(s.split("/")[-2:])
    audio_mapping = {get_filename(audio_file): audio_file for audio_file in audio_files}
    return audio_mapping

def parse_meta_data(meta_file: str):
    df = pd.read_csv(meta_file, sep="|")
    return df

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    subset = args.subset
    
    os.makedirs(manifest_dir, exist_ok=True)

    if subset == "medium":
        meta_file = f"{dataset_dir}/data/meta/train_medium.csv"
        audio_root = f"{dataset_dir}/data/audio/m_additional"
    elif subset == "small":
        meta_file = f"{dataset_dir}/data/meta/train_small.csv"
        audio_root = f"{dataset_dir}/data/audio/s"
    elif subset == "train":
        meta_file = f"{dataset_dir}/data/meta/train.csv"
        audio_root = f"{dataset_dir}/data/audio/*"
    else:
        meta_file = f"{dataset_dir}/data/meta/{subset}.csv"
        audio_root = f"{dataset_dir}/data/audio/{subset}"
    
    meta_info = parse_meta_data(meta_file)    
    audio_mapping = generate_audio_mapping(audio_root=audio_root)
    
    cuts = []
    num_cuts = 0
    for _, row in meta_info.iterrows():
        path = row["wav_filename"]
        id = path.replace(".wav", "")
        
        if path not in audio_mapping:
            logging.info(f"no file found for {path}")
            continue
        else:
            audio_file = audio_mapping[path]
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
        
        cuts.append(cut)
        num_cuts += 1
        if num_cuts % 100 == 0:
            logging.info(f"Processed {num_cuts} cuts until now.")
    
    logging.info(f"After filtering, a total of {len(cuts)} valid samples.")
    cuts = CutSet.from_cuts(cuts)
    cuts = cuts.resample(16000)
    
    manifest_output_dir = manifest_dir + "/" + f"spgispeech_cuts_{subset}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()