import argparse
import csv
import glob
import logging
import os

import torch
from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.audio.utils import AudioLoadingError
from lhotse.supervision import SupervisionSegment

from icefall.utils import str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="download/MTG",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/mtg_manifest",
    )
    
    parser.add_argument(
        "--truncate",
        type=str2bool,
        default=True,
        help="If True, we truncate the audio into windows"
    )
    
    parser.add_argument(
        "--window-size",
        type=float,
        default=10,
        help="The window size for truncating"
    )
    
    parser.add_argument(
        "--audio-format",
        type=str,
        default="mp3",
        choices=["mp3", "wav"],
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    truncate = args.truncate
    window_size = args.window_size
    audio_format = args.audio_format
    
    os.makedirs(manifest_dir, exist_ok=True)
    
    audio_files = glob.glob(f"{dataset_dir}/*/*.{audio_format}")
    logging.info(f"Found a total of {len(audio_files)} {audio_format} files")        
    
    all_cuts = []
    num_cuts = 0
    
    if truncate:
        manifest_output_dir = manifest_dir + "/" + f"mtg_cuts_{window_size}s.jsonl.gz"
    else:
        manifest_output_dir = manifest_dir + "/" + f"mtg_cuts_all.jsonl.gz"
    
    if os.path.exists(manifest_output_dir):
        logging.info(f"Maniest already created: {manifest_output_dir}! Please consider removing it.")
        return
    
    for audio_file in audio_files:
        try:
            recording = Recording.from_file(audio_file)
        except AudioLoadingError as e:
            logging.warning(f"Failed to load {audio_file}: {e}")
            continue
        cut_id = audio_file.split("/")[-1].replace(f".{audio_format}", "")
        cut = MonoCut(
            id=cut_id,
            start=0,
            duration=recording.duration,
            channel=0, # we always use the first channel
            recording=recording,
        )
        if truncate:
            # truncate the cut into windows
            cuts = cut.cut_into_windows(window_size)
            for window_cut in cuts:
                sup = SupervisionSegment(
                    id=window_cut.id,
                    recording_id=cut.recording.id,
                    start=0.0, # always zero
                    channel=0,
                    duration=window_cut.duration,
                )
                sup.audio_event = "music"
                window_cut.supervisions = [sup]
                all_cuts.append(window_cut)
                num_cuts += 1
                if num_cuts % 100 == 0:
                    logging.info(f"Processed {num_cuts} cuts until now.")
        else:
            sup = SupervisionSegment(
                id=cut_id,
                recording_id=cut.recording.id,
                start=0.0,
                channel=0,
                duration=cut.duration,
            )
            sup.audio_event = "music"
            cut.supervisions = [sup]
            all_cuts.append(cut)
            num_cuts += 1
        
            if num_cuts % 100 == 0:
                logging.info(f"Processed {num_cuts} cuts until now.")
    
    logging.info(f"After filtering, a total of {len(all_cuts)} valid samples.")
    all_cuts = CutSet.from_cuts(all_cuts)
    all_cuts = all_cuts.resample(16000) # original dataset is 44.1kHz
    
    logging.info(f"Storing the manifest to {manifest_output_dir}")
    all_cuts.to_jsonl(manifest_output_dir)
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()