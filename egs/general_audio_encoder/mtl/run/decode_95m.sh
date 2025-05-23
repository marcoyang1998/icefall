#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

output_ds=2
post_output_ds=1

export CUDA_VISIBLE_DEVICES="1"
# for epoch in $(seq 50 -1 49); do
#     for avg in $(seq $(( epoch - 2)) -1 $(( epoch - 10))); do
# for m in greedy_search modified_beam_search; do
#     for epoch in $(seq 51 1 60); do
#         for avg in $(seq $(( epoch - 7 )) 1 $(( epoch - 1 ))); do
for m in greedy_search modified_beam_search; do
    for epoch in 7; do
        for avg in 2; do
            python zipformer_audio_encoder/decode.py \
                --epoch $epoch \
                --avg $avg \
                --use-averaged-model 1 \
                --num-encoder-layers 2,2,3,4,3,2 \
                --feedforward-dim 512,768,1024,1536,1024,768 \
                --encoder-dim 192,256,448,768,448,192 \
                --encoder-unmasked-dim 192,192,256,256,256,192 \
                --manifest-dir data/fbank_librispeech \
                --output-downsampling-factor $output_ds \
                --post-encoder-downsampling-factor $post_output_ds \
                --joiner-dim 768 --decoder-dim 768 \
                --on-the-fly-feats 1 \
                --exp-dir zipformer_audio_encoder_finetune/exp-finetune-ls-960h-lr-0.02-causal-0-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-time-warp-80-from-hubert-large-layer-21-normalized-mvq-lh-large-shar-300k \
                --decoding-method $m \
                --beam-size 8 \
                --max-duration 1000
        done
    done
done