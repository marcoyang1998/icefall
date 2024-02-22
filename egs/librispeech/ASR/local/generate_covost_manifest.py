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

def parse_tsv(tsv_file):
    mapping = {}
    with open(tsv_file, "r") as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            filename = row[0]
            translation= row[2]
            mapping[filename] = translation
    return mapping

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--translation-folder",
        type=str,
        default="download/covost/translations",
    )

    parser.add_argument(
        "--input-manifest",
        type=str,
        help="the path to the common voice manifest",
        required=True,
    )

    parser.add_argument(
        "--input-language",
        type=str,
        default="en",
    )

    parser.add_argument(
        "--output-language",
        type=str,
        default="zh-CN"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "dev"]
    )

    parser.add_argument(
        "--output-manifest-dir",
        type=str,
        help="The folder to store the output manifest",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    translation_folder = args.translation_folder
    input_manifest = args.input_manifest
    output_manifest_dir = args.output_manifest_dir
    input_language = args.input_language
    split = args.split
    output_language = args.output_language

    translation_csv = f"{translation_folder}/{input_language}_{output_language}/covost_v2.{input_language}_{output_language}.{split}.tsv"
    tsv_file = parse_tsv(translation_csv)

    cv_cuts = load_manifest_lazy(input_manifest)

    new_cuts = []
    for i, cut in enumerate(cv_cuts):
        filename = cut.id.split('/')[-1]
        if filename not in tsv_file:
            logging.info(f"Skipping {filename}")
            continue
        cut.translation = tsv_file[filename] # get the translation
        new_cuts.append(cut)

        if i % 100 == 0 and i:
            logging.info(f"Processed {i} cuts until now.")

    cuts = CutSet.from_cuts(new_cuts)

    manifest_output_dir = output_manifest_dir + "/" + f"cuts_covost_{input_language}_{output_language}_{split}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()