import argparse

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
        "--dataset-dir",
        type=str,
        default="downloads/vox1_test_wav"
    )
    
    parser.add_argument(
        "--manifest-output-dir",
        type=str,
        default="data/fbank/cuts_vox1_test.jsonl.gz"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"]
    )
    
    parser.add_argument(
        "--feat-output-dir",
        type=str,
        default="data/fbank",
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_output_dir = args.manifest_output_dir
    feat_output_dir = args.feat_output_dir
    
    num_jobs = 15
    num_mel_bins = 80
    
    if "vox1" in dataset_dir:
        audio_files = glob.glob(f"{dataset_dir}/*/*/*/*.wav")
        dataset = "vox1"
    elif "vox2" in dataset_dir:
        # need to first convert w4a to wav
        # reference: https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830
        audio_files = glob.glob(f"{dataset_dir}/*/*/*/*.wav")
        dataset = "vox2"
    else:
        raise ValueError()
    
    new_cuts = []
    for i, audio in enumerate(audio_files):
        cut_id = '/'.join(audio.split('/')[-3:])
        recording = Recording.from_file(audio, cut_id)
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
        )
        new_cuts.append(cut)
        
        if i % 100 == 0 and i:
            logging.info(f"Processed {i} cuts until now.")
        
    cuts = CutSet.from_cuts(new_cuts)
    
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
    
    logging.info(f"Computing fbank features for {dataset}")
    with get_executor() as ex:
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{feat_output_dir}/{dataset}_{args.split}_feats",
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
    
    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)
    
def update_manifest(manifest, output_manifest):
    cuts = load_manifest(manifest)
    def _add_speaker(c):
        c.supervisions = [
            SupervisionSegment(
                id=c.id,
                recording_id=c.recording.id,
                start=0.0,
                channel=0,
                duration=c.duration,
                speaker=c.id.split('/')[0]
            )
        ]
        return c
        
    cuts.map(_add_speaker)
    logging.info(f"Saving to {output_manifest}")
    cuts.to_jsonl(output_manifest)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    # main()
    manifest = 'data/fbank/cuts_vox1.jsonl.gz'
    output_manifest = 'data/fbank/cuts_vox1.jsonl.gz'
    update_manifest(manifest, output_manifest)