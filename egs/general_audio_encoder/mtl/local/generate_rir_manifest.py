import argparse
import csv
import logging

from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default="download"
    )

    parser.add_argument("--rir-list-file", type=str, default="download/RIRS_NOISES/real_rirs_isotropic_noises/rir_list")
    
    parser.add_argument(
        "--output-folder", type=str, default="data/rir"
    )
    
    return parser.parse_args()

def parse_txt(txt):
    # return all the rir wav files in the txt file
    data = []
    with open(txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            rir_wav = parts[-1]
            data.append(rir_wav)
    return data

def create_manifest(args):
    rir_list_file = args.rir_list_file
    dataset_folder = args.dataset_folder
    logging.info(f"Creating manifest from {rir_list_file}")
    
    data = parse_txt(rir_list_file)
    
    cuts = []
    for wav_file in data:
        cut_id = wav_file.replace(".wav", "")
        wav_file = dataset_folder + "/" + wav_file
        recording = Recording.from_file(wav_file)
        
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=recording.duration,
            channel=0, # we only keep the first channel
            recording=recording,
        )
        cuts.append(cut)
    
    cuts = CutSet.from_cuts(cuts)
    import pdb; pdb.set_trace()
    output_manifest = args.output_folder + "/rir_cuts.jsonl.gz"
    logging.info(f"Saving the manifest to {output_manifest}")
    cuts.to_jsonl(output_manifest)
    

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    create_manifest(args=args)