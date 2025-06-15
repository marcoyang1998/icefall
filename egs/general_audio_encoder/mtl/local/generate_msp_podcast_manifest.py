import argparse
import csv
import json
import logging
import os

import torch
import lhotse
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
        "--msp-dataset",
        type=str,
        help="Path to the msp-podcast dataset",
        default="download/msp-podcast",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/msp_podcast_manifest",
    )
    
    return parser.parse_args()
    
def get_transcript(filename):
    with open(filename, "r") as f:
        data = f.read().strip()
    return data
    
def main():
    args = get_parser()
    
    msp_dataset = args.msp_dataset
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    audio_folder = f"{msp_dataset}/Audios"
    transcript_folder = f"{msp_dataset}/Transcripts"
    label_file = "download/msp-podcast/Labels/labels_consensus.json"
    
    with open(label_file, "r") as f:
        metadata = json.load(f)
        
    subsets = ["Train", "Development", "Test1", "Test2"]
    new_cuts = {subset: [] for subset in subsets}
    
    num_cuts = 0
    for k, info in metadata.items():
        cut_id = k.replace(".wav", "") 
        audio_file = audio_folder + "/" + k
        recording = Recording.from_file(audio_file, cut_id)
        
        transcript_file = transcript_folder + "/" + cut_id + ".txt"
        text = get_transcript(transcript_file)
        
        emo_class = info["EmoClass"]
        spkr = info["SpkrID"]
        gender = info["Gender"]
        split = info["Split_Set"]
        
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
            text=text,
            gender=gender,
            speaker=spkr,
        )
        supervision.emotion = emo_class
        cut.supervisions = [supervision]
        
        new_cuts[split].append(cut)
        
        num_cuts += 1
        if num_cuts % 100 == 0 and num_cuts:
            logging.info(f"Processed {num_cuts} cuts until now.")
    
    import pdb; pdb.set_trace()
    for subset in subsets:
        cur_cuts = CutSet.from_cuts(new_cuts[subset])
        manifest_output_dir = output_dir + "/" + f"msp_podcast_cuts_{subset}.jsonl.gz"
        
        logging.info(f"Storing the manifest to {manifest_output_dir}")
        cur_cuts.to_jsonl(manifest_output_dir)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()    
    
