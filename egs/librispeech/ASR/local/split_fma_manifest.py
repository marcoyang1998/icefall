import argparse
import csv
import os

import logging
from generate_fma_manifest import load_csv
from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
import pandas as pd
from argparse import ArgumentParser

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="downloads/fma",
        help="Where is the FMA dataset",
    )

    
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["training", "test", "validation"],
        help="Which split to extract"
    )

    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="Where to store the output manifest"
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    manifest = args.manifest
    dataset_dir = args.dataset_dir
    split = args.split

    cuts_full = load_manifest(manifest)

    tracks = load_csv(dataset_dir + '/data/fma_metadata/tracks.csv')
    subset = tracks[tracks['set', 'split'] == args.split]
    indexes = subset.index.tolist()

    def filter_cut(c):
        track_id = int(c.id.split("_")[-1])
        # if tracks.iloc[track_id]["set", "split"] == args.split:
        #     return True
        if track_id in indexes:
            return True
        return False

    cuts_filtered = cuts_full.filter(filter_cut)
    logging.info(f"Before filtering: {len(cuts_full)} cuts")
    logging.info(f"After filtering: {len(cuts_filtered)} cuts")

    logging.info(f"Saving the output manifest to {args.output_manifest}")
    cuts_filtered.to_jsonl(args.output_manifest)
    


if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
