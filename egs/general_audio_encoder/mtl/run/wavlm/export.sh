#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
wavlm_version=base

python wavlm_pretrain/export.py \
    --iter 300000 \
    --avg 4 \
    --use-averaged-model 1 \
    --wavlm-version $wavlm_version \
    --exp-dir wavlm_pretrain/exp-wavlm-base-adamw-lr-5e-4-warm-32000-ls-full-mask-prob-0.65-musan-1-rir-0-hubert-large-layer-21-libri-mvq-cb16-shar \
    --num-codebooks $num_codebookds