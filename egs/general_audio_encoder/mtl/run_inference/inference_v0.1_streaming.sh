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
    --audio models/v0.1/672-122797-0000.flac


# Expected output:
# tensor([[[-0.4407, -0.2123, -0.2616,  ..., -0.0032, -0.0101, -0.0093],
#          [-0.0862, -0.0564, -0.1177,  ..., -0.0032, -0.0101, -0.0093],
#          [-0.1066,  0.0658, -0.0561,  ..., -0.0032, -0.0101, -0.0093],
#          ...,
#          [-0.2710, -0.3413,  0.0121,  ..., -0.0533,  0.0551,  0.0114],
#          [-0.0136, -0.4045, -0.2446,  ..., -0.0533,  0.0551,  0.0114],
#          [ 0.1221, -0.3405, -0.4357,  ..., -0.0533,  0.0551,  0.0114]]],
#        device='cuda:0', grad_fn=<CatBackward0>)