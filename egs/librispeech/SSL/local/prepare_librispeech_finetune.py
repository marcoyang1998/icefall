#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Xiaoyu Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import glob
from pathlib import Path
from typing import Optional

import torch
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--subset",
        type=str,
        default="10h"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default="data/manifests"
    )
    
    return parser.parse_args()

def main(args):
    subset = args.subset
    dataset_root = "download/librispeech_finetuning"
    if subset == "10h":
        subsets = ["1h", "9h"]
    else:
        subsets = [subset]
    
    # get recording set
    wav_files_all = []
    txt_files_all = []
    texts_all = {}
    for split in subsets:
        if split == "1h":
            wav_files = glob.glob(f"{dataset_root}/{split}/*/*/*/*/*.flac")
            txt_files = glob.glob(f"{dataset_root}/{split}/*/*/*/*/*.txt")
        elif split == "9h":
            wav_files = glob.glob(f"{dataset_root}/{split}/*/*/*/*.flac")
            txt_files = glob.glob(f"{dataset_root}/{split}/*/*/*/*.txt")
        
        wav_files_all += wav_files
        txt_files_all += txt_files
    
    logging.info(f"Finding a total of {len(wav_files_all)} audio")
    # prepare transcripts
    for txt in txt_files_all:
        with open(txt, "r") as f:
            data = f.readlines()
        for line in data:
            id, text = line.strip().split(" ", 1)
            texts_all[id] = text
            
    assert len(texts_all) == len(wav_files_all)
    
    all_cuts = []
    for wav in wav_files_all:
        cut_id = wav.split("/")[-1].split(".")[0]
        recording = Recording.from_file(wav)
        supervision = SupervisionSegment(
            id=cut_id,
            text=texts_all[cut_id],
            recording_id=recording.id,
            start=0.0,
            channel=0,
            duration=recording.duration,
        )
        cut = MonoCut(
            id=cut_id,
            recording=recording,
            supervisions=[supervision],
            start=0.0,
            duration=recording.duration,
            channel=0,
        )
        all_cuts.append(cut)
        
    all_cuts = CutSet.from_cuts(all_cuts)
    output_manifest = f"{args.output_folder}/librispeech_finetuning_cuts_{subset}.jsonl.gz"
    logging.info(f"Saving the cuts to {output_manifest}")
    all_cuts.to_jsonl(output_manifest)
        
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    main(args)