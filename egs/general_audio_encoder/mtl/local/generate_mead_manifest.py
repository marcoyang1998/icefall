import argparse
import csv
import logging
import glob
import os

import torch
from lhotse.audio.utils import AudioLoadingError
from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to the MEAD dataset",
        default="download/MEAD",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mead_manifest",
    )
    
    return parser.parse_args()

def main():
    args = get_parser()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    speakers = os.listdir(dataset_dir)
    
    new_cuts = []
    num_cuts = 0
    for spkr in speakers:
        audio_folder = os.path.join(dataset_dir, spkr, "audio")
        
        audios = glob.glob(f"{audio_folder}/*/*/*.m4a")
        for audio in audios:
            emotion, level, uttid = audio.split("/")[-3:]
            cut_id = "-".join([spkr, emotion, level, uttid.replace(".m4a", "")])
            
            try:
                recording = Recording.from_file(audio, cut_id)
            except AudioLoadingError:
                logging.info(f"Skipping {audio} because the file is incomplete")
                continue
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
                speaker=spkr,
                duration=cut.duration,
                text="This is a dummy text."
            )
            supervision.emotion = emotion
            
            cut.supervisions = [supervision]
            new_cuts.append(cut)
            
            num_cuts += 1
            if num_cuts % 100 == 0 and num_cuts:
                logging.info(f"Processed {num_cuts} cuts until now.")
                
    import pdb; pdb.set_trace()
    cur_cuts = CutSet.from_cuts(new_cuts)
    manifest_output_dir = output_dir + "/" + "mead_cuts_all.jsonl.gz"
    
    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cur_cuts.to_jsonl(manifest_output_dir)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()    