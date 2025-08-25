#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_cb=0

causal=0
num_layers=18
num_heads=16
encoder_dim=1024
use_flash_attention=1

python ./transformer/export.py \
    --iter 400000 \
    --avg 4 \
    --use-averaged-model 1 \
    --exp-dir transformer/exp-300m-transformer-causal-0-adamw-wd-0.01-lr-1.5e-3-cosine-scheduler-warmup-32000-lh-large-mask-ratio-1.0-musan-1-rir-0-hubert-large-layer-21-libri-mvq-cb16-shar \
    --num-layers $num_layers \
    --num-heads $num_heads \
    --encoder-dim $encoder_dim \
    --num-codebooks 0 \
    --use-flash-attention $use_flash_attention \
    --causal $causal
