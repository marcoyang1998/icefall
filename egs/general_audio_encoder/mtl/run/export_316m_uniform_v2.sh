#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=0
output_ds=2
teacher_frame_ratio=1

python zipformer_audio_encoder/export.py \
    --iter 400000 \
    --avg 4 \
    --exp-dir zipformer_audio_encoder/exp-316M-uniform-v2-zipformer-out-ds-2-lh-large-giga-xl-voxpopuli-1-as-full-x2-all-audio-w2v2-mask-p-0.65-l-10-cha-mask-p-0.25-l-20-musan-p-0.5-min-snr-10-multi-mvq-wavlm-all-wavlm-large-cb16-1.0-dasheng-cb8-0.1-md300 \
    --output-downsampling-factor $output_ds \
    --use-averaged-model 1 \
    --causal 0 \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,2,3,1,1,1 \
    --feedforward-dim 3072,3072,3072,3072,3072,3072,3072 \
    --encoder-dim 1024,1024,1024,1024,1024,1024,1024 \
    --encoder-unmasked-dim 512,512,512,512,512,512,512 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --num-codebooks $num_codebooks \
    --teacher-frame-ratio $teacher_frame_ratio \
    --distillation-delta $delta

