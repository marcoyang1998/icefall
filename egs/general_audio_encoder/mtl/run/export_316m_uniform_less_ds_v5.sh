#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=0
output_ds=1
teacher_frame_ratio=1

python zipformer_audio_encoder/export.py \
    --iter 400000 \
    --avg 4 \
    --exp-dir zipformer_audio_encoder/exp-316M-uniform-less-ds-v5-out-ds-1-zipformer-lh-large-giga-xl-emo-0-vox-1-lr-hours-30000-lr-batches-7500-w2v2-mask-p-0.5-mask-l-10-cha-mask-p-0.25-l-15-musan-1-p-0.5-min-snr-10-max-snr-20-rir-0-wavlm-large-layer-21-libri-mvq-cb-16-shar-md-400-16-gpus \
    --output-downsampling-factor $output_ds \
    --use-averaged-model 1 \
    --causal 0 \
    --downsampling-factor 1,2,4,2,1 \
    --num-encoder-layers 1,2,5,2,1 \
    --feedforward-dim 3072,3072,3072,3072,3072 \
    --encoder-dim 1024,1024,1024,1024,1024 \
    --encoder-unmasked-dim 512,512,512,512,512 \
    --cnn-module-kernel 31,31,15,31,31 \
    --num-heads 8,8,8,8,8 \
    --num-codebooks $num_codebooks \
    --teacher-frame-ratio $teacher_frame_ratio \
    --distillation-delta $delta

