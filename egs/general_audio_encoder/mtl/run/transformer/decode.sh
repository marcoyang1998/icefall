#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH


# model related
causal=0
num_layers=18
num_heads=16
encoder_dim=1024
use_flash_attention=1

export CUDA_VISIBLE_DEVICES="1"

exp_dir=exp-finetune-300M-ls-960--wd-0.01-sched-eden-lr-1e-3-causal-0-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-from-hubert-large-mvq-cb16-with-musan-no-rir-400k

for m in greedy_search modified_beam_search; do
    for epoch in 50; do
        for avg in 30; do
            python transformer/decode.py \
                --epoch $epoch \
                --avg $avg \
                --use-averaged-model 1 \
                --num-layers $num_layers \
                --num-heads $num_heads \
                --encoder-dim $encoder_dim \
                --use-flash-attention $use_flash_attention \
                --causal $causal \
                --manifest-dir data/fbank_librispeech \
                --joiner-dim 512 --decoder-dim 512 \
                --on-the-fly-feats 1 \
                --exp-dir transformer_finetune/${exp_dir} \
                --decoding-method $m \
                --beam-size 8 \
                --max-duration 1000
        done
    done
done