#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH


num_layers=24
num_heads=16
encoder_dim=1024
use_flash_attention=1
subsampling_factor=4
causal=1

exp_dir=exp-finetune-420m-causal-1-sub-4-ls-960--lr-3e-5-cosine-scheduler-warmup-12000-causal-1-freeze-encoder-0-freeze--1-step-encoder-lr-scale-1.0-from-hubert-large-mvq-cb16-delta-6-lh-large-giga-xl-pt-attn-drop-0.1-cosine-sched-with-musan-no-rir-400k

# for m in greedy_search modified_beam_search; do
#     for epoch in 30; do
#         for avg in $(seq 15 -1 10); do
for m in greedy_search; do
    for epoch in 999; do
        for avg in 1; do
            python transformer/decode.py \
                --epoch $epoch \
                --avg $avg \
                --use-averaged-model 0 \
                --causal $causal \
                --subsampling-factor $subsampling_factor \
                --num-layers $num_layers \
                --num-heads $num_heads \
                --encoder-dim $encoder_dim \
                --use-flash-attention $use_flash_attention \
                --manifest-dir data/fbank_librispeech \
                --on-the-fly-feats 1 \
                --exp-dir transformer_finetune/$exp_dir \
                --decoding-method $m \
                --beam-size 8 \
                --max-duration 400
        done
    done
done