import argparse
import csv

import torch
import torchaudio
import logging
import glob
from lhotse import load_manifest, CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment
from argparse import ArgumentParser

from icefall.utils import get_executor, str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def parse_clotho(csv_file: str):

    data = {} # a dict, key is the filename, value is a list of audio captions
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            filename = row[0]
            captions = row[1:]
            assert len(captions) == 5
            filename = filename.rsplit(".", 1)[0]
            data[filename] = captions
    return data

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--clotho-dataset",
        type=str,
        help="Path to the clotho dataset",
        default="download/clotho-dataset/data",
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--clotho-csv",
        type=str,
        help="The path to the audio captions",
        required=True,
    )

    parser.add_argument(
        "--fbank-dir",
        type=str,
        default="data/fbank_clotho",
    )

    parser.add_argument(
        "--clotho-version",
        type=str,
        default="2.1",
        help="The version of the clotho dataset"
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    clotho_dataset = args.clotho_dataset
    clotho_csv = args.clotho_csv
    split = args.split
    fbank_dir = args.fbank_dir
    version = args.clotho_version

    num_jobs = 15
    num_mel_bins = 80

    clotho_captions = parse_clotho(clotho_csv)

    wav_files = glob.glob(f"{clotho_dataset}/{split}/*.wav")
    logging.info(f"Find a total of {len(wav_files)} wave files")

    new_cuts = []

    for i, audio in enumerate(wav_files):
        filename = audio.split('/')[-1].rsplit(".", 1)[0]
        assert filename in clotho_captions, filename
        captions = clotho_captions[filename]
        cut_id = filename.replace(" ", "__")
        recording = Recording.from_file(audio, cut_id)
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
        )
        supervision = SupervisionSegment(
            id=cut_id,
            recording_id=cut.recording.id,
            start=0.0,
            channel=0,
            duration=cut.duration,
        )

        supervision.audio_captions = ";;".join(captions)

        cut.supervisions = [supervision]
        new_cuts.append(cut)

        if i % 100 == 0 and i:
            logging.info(f"Processed {i} cuts until now.")
    
    cuts = CutSet.from_cuts(new_cuts)
    cuts = cuts.resample(16000)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    logging.info(f"Computing fbank features for {split}")
    with get_executor() as ex:
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{fbank_dir}/{split}_feats_v{version}",
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

    manifest_output_dir = fbank_dir + "/" + f"cuts_clotho_{split}.v{version}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)


if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()