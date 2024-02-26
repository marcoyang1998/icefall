import argparse
import csv
import os

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
from lhotse import RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from argparse import ArgumentParser

from icefall.utils import get_executor, str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def parse_tsv(tsv_file):
    # the path of the validated.tsv
    mapping = {}
    with open(tsv_file, "r") as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            spkr_id, filename, sentence = row[0:3]
            mapping[filename] = [sentence, spkr_id]
    return mapping

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-language",
        type=str,
        default="en"
    )

    parser.add_argument(
        "--fbank-dir",
        type=str,
        default="data/fbank_covost2",
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default="download/covost/"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "dev", "other"],
        required=True,
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    input_language = args.input_language
    fbank_dir = args.fbank_dir
    split = args.split
    root_dir = args.root_dir

    tsv_file = f"{root_dir}/{input_language}/{split}.tsv"
    tsv_file = parse_tsv(tsv_file)

    recordings = []
    supervisions = []
    num_cuts = 0
    for filename in tsv_file:
        full_path = f"{root_dir}/{input_language}/clips/{filename}"
        sentence, spkr_id = tsv_file[filename]
        cut_id = full_path

        # There are some broken files in common voice
        if os.path.getsize(full_path) == 0:
            logging.info(f"Skipping {full_path}")
            continue

        recording = Recording.from_file(full_path, cut_id)
        recordings.append(recording)
        
        supervision = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0.0,
            channel=0,
            text=sentence,
            speaker=spkr_id,
            language=input_language,
            duration=recording.duration,
        )
        supervisions.append(supervision)

        num_cuts += 1
        if num_cuts % 100 == 0:
            logging.info(f"Processed {num_cuts} cuts until now.")

    recordings = RecordingSet.from_recordings(recordings)
    recordings = recordings.resample(16000)
    supervisions = SupervisionSet.from_segments(supervisions)
    
    # Fix any missing recordings/supervisions.
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    cuts = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions,
    )

    num_jobs = 15
    num_mel_bins = 80
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    logging.info(f"Computing fbank features for {split}")
    with get_executor() as ex:
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{fbank_dir}/{input_language}_{split}_feats",
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

    manifest_output_dir = fbank_dir + "/" + f"cuts_covost2_{input_language}_{split}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()