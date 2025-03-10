import argparse
import csv
import logging

from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def is_cut_long(c):
    return c.duration > 5.0

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--csv-file", type=str, default="download/noise_rir/noise.csv")
    
    parser.add_argument(
        "--output-folder", type=str, default="data/noise"
    )
    
    return parser.parse_args()

def parse_csv(csv_file):
    data = []
    with open(csv_file, "r") as fin:
        reader = csv.reader(fin, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            data.append([row[0], row[2]])
    return data

def create_manifest(args):
    csv_file = args.csv_file
    
    data = parse_csv(csv_file)
    
    cuts = []
    for item in data:
        cut_id, audio_file = item
        recording = Recording.from_file(audio_file)
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
        )
        cuts.append(cut)
    
    cuts = CutSet.from_cuts(cuts)
    cuts = cuts.cut_into_windows(10.0).filter(is_cut_long)
    output_manifest = args.output_folder + "/noise_cuts.jsonl.gz"
    logging.info(f"Saving the manifest to {output_manifest}")
    cuts.to_jsonl(output_manifest)
    

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    create_manifest(args=args)