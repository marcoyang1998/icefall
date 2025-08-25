#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=0
output_ds=1
teacher_frame_ratio=1

python zipformer_audio_encoder/export.py \
    --iter 100000 \
    --avg 2 \
    --exp-dir zipformer_audio_encoder/exp-96M-uniform-out-ds-1-zipformer-lh-large-giga-xl-vox-1-lr-batches-7500-lr-hours-30000-w2v2-mask-p-0.5-l-10-cha-mask-p-0.25-l-15-batch-mix-p-0.2-min-snr--5-p-noise-0.1-wavlm-large-layer-21-libri-mvq-cb-16-shar-md-400-norm-fbank-0 \
    --output-downsampling-factor $output_ds \
    --use-averaged-model 1 \
    --causal 0 \
    --num-encoder-layers 1,1,2,2,2,1 \
    --feedforward-dim 1536,1536,1536,1536,1536,1536 \
    --encoder-dim 640,640,640,640,640,640 \
    --encoder-unmasked-dim 256,256,256,256,256,256 \
    --chunk-size 8,16,32,64,-1 \
    --left-context-frames 64,128,-1 \
    --num-codebooks $num_codebooks \
    --teacher-frame-ratio $teacher_frame_ratio \
    --distillation-delta $delta

