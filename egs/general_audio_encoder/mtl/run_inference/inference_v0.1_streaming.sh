#!/usr/bin/env bash

icefall_root=$(realpath ../../..)
export PYTHONPATH=${icefall_root}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES="0"

python zipformer_inference/inference_model_streaming.py \
    --ckpt-path models/v0.1/iter-100000-avg-6.pt \
    --causal 1 \
    --chunk 8 \
    --left-context-frames 256 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --audio models/v0.1/audioset.wav