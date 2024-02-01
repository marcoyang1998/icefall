import argparse
import csv

import torch
import torchaudio
import logging
import glob
from lhotse import load_manifest_lazy, CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment
from lhotse.utils import uuid4
from argparse import ArgumentParser

from copy import deepcopy
from icefall.utils import get_executor, str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    input_manifest = args.input_manifest
    output_manifest = args.output_manifest
    
    cuts = load_manifest_lazy(input_manifest)
    
    new_cuts = []
    for i, c in enumerate(cuts):
        captions = c.supervisions[0].audio_captions.split(';;')
        for j, caption in enumerate(captions):
            new_cut = deepcopy(c)
            new_cut.id = c.id + f"_{j}"
            new_cut.supervisions[0].audio_captions = caption
            new_cuts.append(new_cut)
            
    logging.info(f"The newly created manfiest has {len(new_cuts)} cuts")
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest)
        

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()