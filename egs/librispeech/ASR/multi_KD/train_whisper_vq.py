#!/usr/bin/env python3
# Copyright 2022 Xiaomi Corporation (Author: Xiaoyu Yang)
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
import copy
import glob
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from icefall import is_module_available

if not is_module_available("multi_quantization"):
    raise ValueError("Please 'pip install multi_quantization' first.")

import multi_quantization as quantization
from kd_datamodule import LibriSpeechKDDataModule
from lhotse import CutSet, load_manifest
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer

from icefall.utils import AttributeDict, setup_logger

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="train-clean-100"
    )


    return parser

class VQ_trainer:
    """
    A wrapper of quantiation.Quantizerz.

    It's responsible for:
        1. extract and save activations from a teacher model.
        2. train quantizer from previous activations.
        3. extract codebook indexes for whole training set.
           Normally this step needs multi GPUs.
    """
    def __init__(self, params: AttributeDict):
        self.params = params

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--vq-dir",
            type=str,
            default="data/vq_fbank"
        )

        # Options about teacher embeddings eatraction.
        parser.add_argument(
            "--embedding-dim",
            type=int,
            default=768,
        )

        parser.add_argument(
            "--embedding-file-path",
            type=str,
            required=True,
        )

        parser.add_argument(
            "--num-codebooks",
            type=int,
            default=8,
            help="""number of codebooks,
            i.e. number of codebook indexes each teacher embedding is compressed.
            """,
        )

        parser.add_argument(
            "--quantizer-file-path",
            type=str,
            default="data/quantizer/librispeech_cb16_quantizer.pt",
            help="Path to the quantizer",
        )

        parser.add_argument(
            "--phase-one-iters",
            type=int,
            default=20000,
            help="The number of steps in phase one training of the quantizer"
        )

        parser.add_argument(
            "--phase-two-iters",
            type=int,
            default=20000,
            help="The number of steps in phase one training of the quantizer"
        )

    def load_quantizer(self, device):
        
        assert self.quantizer_file_path.exists()
        quantizer = quantization.Quantizer(
            dim=self.params.embedding_dim,
            num_codebooks=self.params.num_codebooks,
            codebook_size=256,
        )
        quantizer.load_state_dict(torch.load(self.quantizer_file_path))

        return quantizer

    def train_quantizer(self):
        
        if self.params.quantizer_file_path.exists():
            logging.info(f"The quantizer already exists at {self.params.quantizer_file_path}")
            return

        logging.info("Start to train quantizer.")
        trainer = quantization.QuantizerTrainer(
            dim=self.params.embedding_dim,
            bytes_per_frame=self.params.num_codebooks,
            device=self.params.device,
            phase_one_iters=self.params.phase_one_iters,
            phase_two_iters=self.params.phase_two_iters,
        )
        train, valid = quantization.read_hdf5_data(self.params.embedding_file_path)
        B = 512  # Minibatch size, this is very arbitrary,
        # it's close to what we used when we tuned this method.

        def minibatch_generator(data: torch.Tensor, repeat: bool):
            assert 3 * B < data.shape[0]
            cur_offset = 0
            while True if repeat else cur_offset + B <= data.shape[0]:
                start = cur_offset % (data.shape[0] + 1 - B)
                end = start + B
                cur_offset += B
                yield data[start:end, :].to(self.params.device).to(dtype=torch.float)

        for x in minibatch_generator(train, repeat=True):
            trainer.step(x)
            if trainer.done():
                break

        quantizer = trainer.get_quantizer()
        torch.save(quantizer.state_dict(), self.params.quantizer_file_path)

    @torch.no_grad()
    def extract_codebook_indexes(self, quantizer):
        
        
        manifest_file_path = self.params.vq_dir / f"librispeech_{params.subset}_cb_{params.num_codebooks}.h5"

        num_cuts = 0
        new_cuts = []

        with NumpyHdf5Writer() as writer:
            for batch_idx, batch in enumerate(self.manifest):
                whisper_embeddings = batch["whisper_embedding"]
                whisper_embedding_lens = batch["whisper_embedding"]
                supervisions = batch["supervisions"]
                cut_list = supervisions["cut"]

                codebook_indexes = quantizer.encode(encoder_embedding)

                assert np.min(codebook_indexes) >= 0
                assert np.max(codebook_indexes) < 256
                assert len(cut_list) == codebook_indexes.shape[0]
                assert all(c.start == 0 for c in supervisions["cut"])

                for idx, cut in enumerate(cut_list):
                    new_cut = MonoCut(
                        id=cut.id,
                        start=cut.start,
                        duration=cut.duration,
                        channel=cut.channel,
                    )
                    new_cut.codebook_indexes = writer.store_array(
                        key=cut.id,
                        value=codebook_indexes[idx][: num_frames[idx]],
                        frame_shift=0.02,
                        temporal_dim=0,
                        start=0,
                    )
                    new_cuts.append(new_cut)
                    num_cuts += 1

                logging.info(f"Processed {num_cuts} so far")


        output_manifest = self.params.vq_dir / f"with_cb_librispeech_{self.params.subset}.jsonl.gz"
        
        logging.info(f"Saving manifest to {output_manifest}")
        CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)


def main(params):
    
    setup_logger(f"{params.vq_dir}/log-vq-extraction")

    logging.info("Training started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    params.device = device

    vq_trainer = VQ_trainer(params)
    
    vq_trainer.train_quantizer()
    import pdb; pdb.set_trace()
    quantizer = vq_trainer.load_quantizer()
    quantizer.to(device)

    vq_trainer.extract_codebook_indexes(quantizer=quantizer)
    
    pass


if __name__=="__main__":

    parser = get_parser()
    VQ_trainer.add_arguments(parser)

    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))

    params.quantizer_file_path = Path(params.quantizer_file_path)
    params.vq_dir = Path(params.vq_dir)

    main(params)

    logging.info(f"Finished")
    # embedding_file = "data/embeddings/whisper-small.en-embeddings-dev-clean-0.h5"
    # train, valid = quantization.read_hdf5_data(embedding_file)
    # print("finished")