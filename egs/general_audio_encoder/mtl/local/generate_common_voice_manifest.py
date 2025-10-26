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
        default="download/cv-corpus-4.0",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/cv_manifest",
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="train"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="en",
    )
    
    return parser

def parse_tsv(tsv_file: str):

    df = pd.read_csv(tsv_file, sep='\t')
    df = df[["path", "sentence", "gender"]]
    return df

def generate_audio_mapping(audio_root: str):
    audio_files = glob.glob(f"{audio_root}/*/*.wav")
    def get_audio_id(s: str):
        filename = s.split("/")[-1]
        return filename.split(".")[0]
    audio_mapping = {get_audio_id(audio_file): audio_file for audio_file in audio_files}
    return audio_mapping

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    subset = args.subset
    language = args.language
    
    os.makedirs(manifest_dir, exist_ok=True)
    
    tsv_file = f"{dataset_dir}/transcript/{language}/{subset}.tsv"
    meta_info = parse_tsv(tsv_file)
    audio_root = f"{dataset_dir}/audio/{language}/{subset}"
    audio_mapping = generate_audio_mapping(audio_root=audio_root)
    
    cuts = []
    num_cuts = 0
    for _, row in tqdm(meta_info.iterrows()):
        path = row["path"]
        id = path.replace(".mp3", "").replace(".wav", "")
        
        if id not in audio_mapping:
            logging.info(f"no file found for {id}")
            continue
        else:
            audio_file = audio_mapping[id]
            # we assume you have already converted the mp3 file to wav
            if audio_file.endswith(".mp3"):
                audio_file = audio_file.replace(".mp3", ".wav")
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
            text=row["sentence"],
            language=language,
            gender=row["gender"],
        )
        cut.supervisions = [supervision]
        
        cuts.append(cut)
        num_cuts += 1
        if num_cuts % 100 == 0:
            logging.info(f"Processed {num_cuts} cuts until now.")
        
    logging.info(f"After filtering, a total of {len(cuts)} valid samples.")
    cuts = CutSet.from_cuts(cuts)
    cuts = cuts.resample(16000)
    
    manifest_output_dir = manifest_dir + "/" + f"commonvoice_{language}_cuts_{subset}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
