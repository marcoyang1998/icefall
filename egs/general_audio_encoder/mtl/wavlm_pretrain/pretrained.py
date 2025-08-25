#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
"""
./wavlm_pretrain/pretrained.py
    --ckpt xx.pt
    --num-codebooks 0

"""


import argparse
import logging
import math

import torch

from train_multi_KD3_shar import add_model_arguments, get_model, get_params

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    add_model_arguments(parser)

    return parser


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    logging.info("Decoding started")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    import pdb; pdb.set_trace()
    state_dict = torch.load(params.ckpt, map_location="cpu")["model"]
    info = model.load_state_dict(state_dict)
    logging.info(info)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    device = torch.device("cuda")
    wavs = [
        torch.randn(16000 * 5),
        torch.randn(16000 * 4)
    ]
    wav_lens = torch.tensor([16000 * 5, 16000 * 4]).to(device)
    
    from torch.nn.utils.rnn import pad_sequence
    wavs = pad_sequence(wavs, batch_first=True).to(device)
    model.to(device)
    model.eval()
    
    import pdb; pdb.set_trace()
    encoder_out, encoder_out_lens = model.forward_encoder(wavs, wav_lens)
    # encoder_out2, encoder_out_lens2 = model.forward_encoder(wavs, wav_lens)
    
    torch.save(wavs, "wavs.pt")
    torch.save(wav_lens, "wav_lens.pt")
    torch.save(encoder_out, "encoder_out_gt.pt")
    torch.save(encoder_out_lens, "encoder_out_lens_gt.pt")
    
    logging.info("Done!")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
