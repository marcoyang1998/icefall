import argparse
import csv
import logging
import os

import torch
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
        help="Path to the msp-podcast dataset",
        default="download/MELD.Raw",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/meld_manifest",
    )
    
    return parser.parse_args()

def parse_file(filename, dataset_dir: str, subset: str = "train"):
    all_data = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["Utterance"]
            speaker = row["Speaker"]
            emotion = row["Emotion"]
            sentiment = row["Sentiment"]
            diag_ID = row["Dialogue_ID"]
            utt_ID = row["Utterance_ID"]
            if subset == "train":
                filename = f"{dataset_dir}/train_splits/dia{diag_ID}_utt{utt_ID}.wav"
            elif subset == "dev":
                filename = f"{dataset_dir}/dev_splits_complete/dia{diag_ID}_utt{utt_ID}.wav"
            else:
                filename = f"{dataset_dir}/output_repeated_splits_test/dia{diag_ID}_utt{utt_ID}.wav"
            item = [filename, text, speaker, emotion, sentiment]
            all_data.append(item)
    return all_data

def main():
    args = get_parser()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    subsets = ["dev", "test", "train"]
    for subset in subsets:
        cuts = []
        num_cuts = 0
        csv_file = f"{dataset_dir}/{subset}_sent_emo.csv"
        meta_data = parse_file(csv_file, dataset_dir, subset=subset)
        
        for item in meta_data:
            filename, text, speaker, emotion, sentiment = item
            cut_id = subset + "_" + filename.split("/")[-1].replace(".wav", "")
        
            if not os.path.exists(filename):
                logging.info(f"Skipping {filename} because it doesn't exist!")
                continue
            recording = Recording.from_file(filename, cut_id)
            cut = MonoCut(
                id=cut_id,
                start=0.0,
                duration=recording.duration,
                channel=0, # we only use the first channel
                recording=recording,
            )
            supervision = SupervisionSegment(
                id=cut_id,
                recording_id=cut.recording.id,
                start=0.0,
                channel=0,
                speaker=speaker,
                duration=cut.duration,
                text=text,
            )
            supervision.emotion = emotion
            supervision.sentiment = sentiment
            cut.supervisions = [supervision]
            cuts.append(cut)
            
            num_cuts += 1
            if num_cuts % 100 == 0 and num_cuts:
                logging.info(f"Processed {num_cuts} cuts until now.")
            
        cur_cuts = CutSet.from_cuts(cuts)
        manifest_output_dir = output_dir + "/" + f"meld_cuts_{subset}.jsonl.gz"
        
        logging.info(f"Storing the manifest to {manifest_output_dir}")
        cur_cuts.to_jsonl(manifest_output_dir)
            
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()    
   
    
