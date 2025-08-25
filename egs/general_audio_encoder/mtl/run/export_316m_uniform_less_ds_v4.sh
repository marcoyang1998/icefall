#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=0
output_ds=1
teacher_frame_ratio=1

python zipformer_audio_encoder/export.py \
    --iter 100000 \
    --avg 2 \
    --exp-dir zipformer_audio_encoder/exp-316M-uniform-less-ds-v4-zipformer-out-ds-1-lh-large-giga-xl-emo-0-vox-1-lr-hours-75000-lr-batches-7500-w2v2-mask-p-0.5-mask-l-10-cha-mask-p-0.25-l-15-batch-mix-p-0.2-min-snr--5-p-noise-0.1-wavlm-large-layer-21-libri-mvq-cb-16-shar-md-400-16-gpus \
    --output-downsampling-factor $output_ds \
    --use-averaged-model 1 \
    --causal 0 \
    --downsampling-factor 1,2,2,4,2,1 \
    --num-encoder-layers 1,2,2,3,2,1 \
    --feedforward-dim 3072,3072,3072,3072,3072,3072 \
    --encoder-dim 1024,1024,1024,1024,1024,1024 \
    --encoder-unmasked-dim 512,512,512,512,512,512 \
    --cnn-module-kernel 31,31,31,15,31,31 \
    --num-heads 8,8,8,8,8,8 \
    --num-codebooks $num_codebooks \
    --teacher-frame-ratio $teacher_frame_ratio \
    --distillation-delta $delta

