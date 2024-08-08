import argparse
import csv
import os

import torch
import torchaudio
import logging
import glob
from lhotse import load_manifest, CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment
from lhotse.audio.utils import AudioLoadingError

from icefall.utils import get_executor

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="downloads/gtzan",
    )

    parser.add_argument(
        "--feat-output-dir",
        type=str,
        default="data/fbank_gtzan",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    feat_output_dir = args.feat_output_dir

    num_jobs = 6
    num_mel_bins = 80

    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

    new_cuts = []
    for genre in genres:
        audios = glob.glob(f"{dataset_dir}/Data/genres_original/{genre}/*.wav")
        assert len(audios) > 0
        logging.info(f"Getting {len(audios)} audios for {genre}")

        for i, audio in enumerate(audios):
            cut_id = f"cut_{genre}_{i}"
            try:
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
                supervision.genre = genre
                cut.supervisions = [supervision]
                new_cuts.append(cut)
            except AudioLoadingError:
                logging.info(f"Audio is broken! Skipping {audio}")
                continue
        logging.info(f"Processed a total of {len(new_cuts)} cuts.")

    cuts = CutSet.from_cuts(new_cuts)
    cuts = cuts.resample(16000)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    logging.info("Computing fbank features for GTZAN")
    with get_executor() as ex:
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{feat_output_dir}/gtzan_feats",
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

    manifest_output_dir = feat_output_dir + "/" + f"cuts_gtzan.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()