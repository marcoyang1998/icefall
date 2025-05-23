#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=0
output_ds=2
teacher_frame_ratio=2

python zipformer_audio_encoder/export.py \
    --iter 224000 \
    --avg 4 \
    --exp-dir zipformer_audio_encoder/exp-96M-zipformer-non-streaming-giga-m-out-ds-2-mask-ratio-1.0-musan-1-rir-0-whisper-turbo-giga-mvq-cb16-shar \
    --output-downsampling-factor $output_ds \
    --use-averaged-model 1 \
    --causal 0 \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --chunk-size 8,16,32,64,-1 \
    --left-context-frames 64,128,-1 \
    --num-codebooks $num_codebooks \
    --teacher-frame-ratio $teacher_frame_ratio \
    --distillation-delta $delta

