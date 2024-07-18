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
import pandas as pd
from argparse import ArgumentParser
import ast

from icefall.utils import get_executor, str2bool

def load_csv(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

def get_audio_path(audio_dir, track_id, extension: str="mp3"):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.

    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'

    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + f'.{extension}')

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="downloads/fma"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="small",
        choices=["small", "medium", "large"]
    )
    
    parser.add_argument(
        "--feat-output-dir",
        type=str,
        default="data/fbank_fma",
    )

    parser.add_argument(
        "--compute-fbank",
        type=str2bool,
        default=True,
        help="If False, only store the manfiest without features",
    )

    parser.add_argument(
        "--extension",
        type=str,
        default="mp3",
        choices=["mp3", "wav"],
        help="Which audio extension to use. The original audio is mp3 format, the wav file is converted from"
        "the original mp3 and resampled to 16kHz"
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    split = args.split
    feat_output_dir = args.feat_output_dir
    extension = args.extension
    
    num_jobs = 15
    num_mel_bins = 80

    # Load metadata
    tracks = load_csv(dataset_dir + '/data/fma_metadata/tracks.csv')

    subset = tracks[tracks['set', 'subset'] <= split]
    audio_folder = os.path.join(dataset_dir, "data", f"fma_{split}")

    new_cuts = []
    for i, (track_id, track) in enumerate(subset.iterrows()):
        audio = get_audio_path(audio_folder, track_id, extension=extension)
        genres = track["track", "genres"]
        genres_all = track["track", "genres_all"]
        
        cut_id = "cut_{}_{:06d}".format(split, track_id)
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
            supervision.genres = genres
            supervision.genres_all = genres_all
            cut.supervisions = [supervision]
            new_cuts.append(cut)
        except AudioLoadingError:
            logging.info(f"Audio is broken! Skipping {audio}")
            continue

        if i % 100 == 0 and i:
            logging.info(f"Processed {i} cuts until now.")

    logging.info(f"Processed a total of {len(new_cuts)} cuts.")

    cuts = CutSet.from_cuts(new_cuts)
    
    def filter_long(c):
        if c.duration > 32:
            print(f"Removing {c.id}: duration {c.duration}")
            return False
        return True
    cuts = cuts.filter(filter_long)

    if not args.compute_fbank:
        manifest_output_dir = feat_output_dir + "/" + f"cuts_fma_{split}_no_feat.jsonl.gz"

        logging.info(f"Storing the manifest to {manifest_output_dir}")
        cuts.to_jsonl(manifest_output_dir)

    else:
        if cuts[0].sampling_rate != 16000:
            cuts = cuts.resample(16000)

        extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

        logging.info(f"Computing fbank features for {split}")
        with get_executor() as ex:
            cuts = cuts.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{feat_output_dir}/fma_{args.split}_feats",
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )

        manifest_output_dir = feat_output_dir + "/" + f"cuts_fma_{split}.jsonl.gz"

        logging.info(f"Storing the manifest to {manifest_output_dir}")
        cuts.to_jsonl(manifest_output_dir)

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()