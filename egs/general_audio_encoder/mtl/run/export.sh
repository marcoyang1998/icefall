#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=3

python zipformer_audio_encoder/export.py \
    --iter 500000 \
    --avg 4 \
    --exp-dir zipformer_audio_encoder/exp-lr-0.04-causal-ls-giga-xl-lh-large-mls-1-extra-zh-en-use-weread-1-as-full-multi-mvq-kd-at-kd-scale-5.0-whisper-all-firered-zh-bucket-sampler-md-320-fix \
    --use-averaged-model 1 \
    --causal 1 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --chunk-size 8,16,32,64,-1 \
    --left-context-frames 64,128,-1 \
    --num-codebooks $num_codebooks \
    --teacher-frame-ratio 2 \
    --distillation-delta $delta

exit


# python zipformer_audio_encoder/export.py \
#     --iter 450000 \
#     --avg 4 \
#     --exp-dir zipformer_audio_encoder/exp-xlarge-lr-0.04-full-en-zh-baoxiang-data-audio-multi-kd-time-mask-ratio-1.0-shar \
#     --use-averaged-model 1 \
#     --causal 1 \
#     --num-encoder-layers 2,2,4,5,4,2 \
#     --feedforward-dim 512,1024,2048,3072,2048,1024 \
#     --encoder-dim 192,384,768,1024,768,384 \
#     --encoder-unmasked-dim 192,256,320,512,320,256 \
#     --chunk-size 8,16,32,64,-1 \
#     --left-context-frames 64,128,-1 \
#     --num-codebooks $num_codebooks \
#     --teacher-frame-ratio 2 \
#     --distillation-delta $delta
