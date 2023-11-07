import argparse
import os
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger
from teachers import WhisperTeacher

import torch
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import DataLoader

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.dataset import SimpleCutSampler, UnsupervisedWaveformDataset, DynamicBucketingSampler
from lhotse.features.io import NumpyHdf5Writer

import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
from typing import Union, Optional

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="name of the manifest, e.g embeddings-dev-clean"
    )

    parser.add_argument(
        "--max-duration",
        type=int,
        default=1000,
    )
    
    parser.add_argument(
        "--whisper-version",
        type=str,
        default="small.en"
    )
    
    return parser

def transcribe(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log-whisper-transcription/log-whisper-transcribe")
        
    device = torch.device("cuda", rank)
    
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.output_manifest.replace(".jsonl.gz", f"-{rank}.jsonl.gz")
    else:
        output_manifest = params.output_manifest

    def remove_short_and_long_utt(c):
        if c.duration < 1.0 or c.duration > 30.0:
            return False
        return True
    
    manifest = manifest.filter(remove_short_and_long_utt)

    dataset = UnsupervisedWaveformDataset(
        manifest
    )
    
    sampler = DynamicBucketingSampler(
        manifest,
        max_duration=params.max_duration,
        shuffle=False,
        drop_last=False,
    )
    
    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=2,
        persistent_workers=False,
    )
    


    new_cuts = []
    num_cuts = 0
    
    # currently only use the encoder of whisper
    logging.info(params)
    model = whisper.load_model(params.whisper_version, device)
    
    for i, batch in enumerate(dl):
        cuts = batch["cuts"]
        audio_input_16khz = batch["audio"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        durations = [c.duration for c in cuts]

        results = model.transcribe_batch(
            audio_input_16khz,
            audio_lens=audio_lens,
            max_duration=max(durations),
            language="en",
        )
        texts = [res.text.strip() for res in results]
        
        for cut, text in zip(cuts, texts):
            cut.supervisions[0].text = text
            cut.whisper_result = text
            new_cuts.append(cut)
            num_cuts += 1

        if i and i % 10 == 0:
            logging.info(f"Cuts processed until now: {num_cuts}")
        
    CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)
    logging.info(f"Saved manifest to {output_manifest}")
        
torch.set_num_threads(1)
torch.set_num_interop_threads(1)        


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    
    nj = params.num_jobs
    cuts = load_manifest(params.input_manifest)
    print(f"Finished loading manifest")
    
    if not os.path.exists(params.output_manifest):
        if nj == 1:
            transcribe(
                rank=0,
                manifest=cuts,
                params=params,    
            )
        else:
            splitted_cuts = cuts.split(num_splits=nj)
            print(f"Finished splitting manifest")
            mp.spawn(transcribe, args=(splitted_cuts, params), nprocs=nj, join=True)
            manifests =  params.output_manifest.replace(".jsonl.gz", "-*.jsonl.gz")
            os.system(f"lhotse combine {manifests} {params.output_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")