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
from argparse import ArgumentParser

from icefall.utils import get_executor, str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--iemocap-dataset",
        type=str,
        help="Path to the iemocap dataset",
        default="download/IEMOCAP_full_release",
    )

    parser.add_argument(
        "--session-id",
        type=int,
        required=True,
    )
    
    parser.add_argument(
        "--fbank-dir",
        type=str,
        default="data/fbank_iemocap",
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.iemocap_dataset
    session_id = args.session_id
    fbank_dir = args.fbank_dir
    
    wav_folder = f"{dataset_dir}/Session{session_id}/dialog/wav"
    label_folder = f"{dataset_dir}/Session{session_id}/dialog/EmoEvaluation"
    
    label_files = sorted(glob.glob(f"{label_folder}/Ses*.txt"))
    
    num_jobs = 4
    num_mel_bins = 80
    
    dataset = {}
    
    for label in label_files:
        with open(label, 'r') as f:
            data = f.readlines()
            
        for line in data:
            # skip lines
            if line[0] != "[": 
                continue
            items = line.strip().split("\t")
            timestamp = items[0].replace("[", "").replace("]", "").split()
            timestamp = [float(timestamp[0]), float(timestamp[2])]
            clip_name = items[1]
            audio_name = clip_name.rsplit("_", 1)[0]
            emotion = items[2]
            audio_name = wav_folder + "/" + f"{audio_name}.wav"
            
            assert os.path.isfile(audio_name)
            assert clip_name not in dataset
            
            dataset[clip_name] = [audio_name, timestamp, emotion]        
    
    logging.info(f"A total of {len(dataset)} clips!")
    
    new_cuts = []
    for i, (cut_id, info) in enumerate(dataset.items()):
        audio_file, timestamp, emotion = info
        recording = Recording.from_file(audio_file, cut_id)
        assert recording.sampling_rate == 16000
        cut = MonoCut(
            id=cut_id,
            start=timestamp[0],
            duration=timestamp[1] - timestamp[0],
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
        supervision.emotion = emotion
        
        cut.supervisions = [supervision]
        new_cuts.append(cut)
        
        if i % 100 == 0 and i:
            logging.info(f"Processed {i} cuts until now.")
    
    cuts = CutSet.from_cuts(new_cuts)
        
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    logging.info(f"Computing fbank features for session {session_id}")
    with get_executor() as ex:
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{fbank_dir}/session{session_id}_feats",
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

    manifest_output_dir = fbank_dir + "/" + f"cuts_iemocap_session{session_id}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()