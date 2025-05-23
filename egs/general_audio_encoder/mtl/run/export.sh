#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=0
output_ds=2
teacher_frame_ratio=2

python zipformer_audio_encoder/export.py \
    --iter 148000 \
    --avg 3 \
    --exp-dir zipformer_audio_encoder/exp-300M-zipformer-non-streaming-lh-large-giga-xl-out-ds-2-mask-ratio-1.0-musan-1-rir-0-hubert-large-layer-21-normalized-mvq-cb16-shar-rerun-2 \
    --output-downsampling-factor $output_ds \
    --use-averaged-model 1 \
    --causal 0 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --chunk-size 8,16,32,64,-1 \
    --left-context-frames 64,128,-1 \
    --num-codebooks $num_codebooks \
    --teacher-frame-ratio $teacher_frame_ratio \
    --distillation-delta $delta

