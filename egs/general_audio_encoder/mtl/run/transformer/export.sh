#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_cb=0

causal=0
num_layers=10
num_heads=8
encoder_dim=768
use_flash_attention=1

python ./transformer/export.py \
    --iter 300000 \
    --avg 4 \
    --use-averaged-model 1 \
    --exp-dir transformer/exp-transformer-causal-0-adam-lr-5e-4-warmup-25000-ls-full-mask-ratio-1.0-musan-1-rir-0-hubert-large-layer-21-libri-mvq-cb16-shar \
    --num-layers $num_layers \
    --num-heads $num_heads \
    --encoder-dim $encoder_dim \
    --num-codebooks 0 \
    --use-flash-attention $use_flash_attention \
    --causal $causal
