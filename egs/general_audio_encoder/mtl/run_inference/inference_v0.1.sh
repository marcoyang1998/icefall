#!/usr/bin/env bash

icefall_root=$(realpath ../../..)
export PYTHONPATH=${icefall_root}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES="0"

python zipformer_inference/inference_model.py \
    --ckpt-path models/v0.1/iter-100000-avg-6.pt \
    --causal 1 \
    --chunk 8 \
    --left-context-frames 256 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --audio models/v0.1/audioset.wav

#####
# Expected output:
#
# torch.Size([1, 251, 1024])
# The topk label are torch.return_types.topk(
# values=tensor([[ 1.5660,  0.2376, -0.9412, -2.4028, -2.4140]], device='cuda:0',
#        grad_fn=<TopkBackward0>),
# indices=tensor([[  0, 465, 137,   1, 450]], device='cuda:0'))
#
#####