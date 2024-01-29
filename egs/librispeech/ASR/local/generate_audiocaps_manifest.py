import argparse
import csv

import torch
import torchaudio
import logging
import glob
from lhotse import (
    load_manifest,
    load_manifest_lazy,
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter
)
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment
from argparse import ArgumentParser

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def parse_audiocaps_csv(csv_file):
    mapping = {}
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            ac_id, ytid, _, caption = row
            if ytid in mapping:
                mapping[ytid].append(caption)
            else:
                mapping[ytid] = [caption]
    logging.info(f"Finished processing {len(mapping)} clips, with {sum([len(v) for k,v in mapping.items()])} captions")
    return mapping

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--audiocaps-dataset",
        type=str,
        default="download/audiocaps"
    )
    
    parser.add_argument(
        "--audioset-manifest",
        type=str,
        default="data/fbank_audioset/cuts_audioset_train_full.jsonl.gz"
    )
    
    parser.add_argument(
        "--audiocaps-split",
        type=str,
        default="train",
        choices=["train", "test", "val"]
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    split = args.audiocaps_split
    audiocaps_dataset = args.audiocaps_dataset
    audioset_manifest_dir = args.audioset_manifest
    
    csv_file = audiocaps_dataset + f'/dataset/{split}.csv'
    captions = parse_audiocaps_csv(csv_file) # a dict from ytid to caption
    
    as_manifest = load_manifest_lazy(audioset_manifest_dir)
    
    new_cuts = []
    for i, c in enumerate(as_manifest):
        wav_id = c.id.split('/')[-1]
        ytid = wav_id.split("_", 1)[0] # the youtube ID
        if ytid not in captions.keys():
            continue
            
        new_cut = c
        new_cut.supervisions[0].audio_captions = ";;".join(captions[ytid])
        
        new_cuts.append(new_cut)
        if i % 100 == 0 and i:
            logging.info(f"Processed {i} cuts until now. {len(new_cuts)} clips are found for audiocaps.")
                
    new_cuts = CutSet.from_cuts(new_cuts)
    manifest_output_dir = f"data/fbank_audiocaps/cuts_audiocaps_{split}.jsonl.gz"
    
    logging.info(f"Saving the processed manifest to {manifest_output_dir}")
    new_cuts.to_jsonl(manifest_output_dir)
    
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()