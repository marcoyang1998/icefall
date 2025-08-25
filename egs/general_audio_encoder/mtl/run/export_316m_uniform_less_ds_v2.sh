#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=0
output_ds=2
teacher_frame_ratio=2

python zipformer_audio_encoder/export.py \
    --iter 400000 \
    --avg 4 \
    --exp-dir zipformer_audio_encoder/exp-316M-uniform-less-ds-v2-zipformer-lh-large-giga-xl-emo-4lr-epochs-1.5-w2v2-mask-prob-0.5-mask-len-10-channel-mask-prob-0.25-len-15-musan-1-rir-0-hubert-large-layer-21-libri-mvq-cb-16-shar-md-400-16-gpus \
    --output-downsampling-factor $output_ds \
    --use-averaged-model 1 \
    --causal 0 \
    --downsampling-factor 1,2,4,4,4,1 \
    --num-encoder-layers 1,2,2,3,2,1 \
    --feedforward-dim 3072,3072,3072,3072,3072,3072 \
    --encoder-dim 1024,1024,1024,1024,1024,1024 \
    --encoder-unmasked-dim 512,512,512,512,512,512 \
    --num-codebooks $num_codebooks \
    --teacher-frame-ratio $teacher_frame_ratio \
    --distillation-delta $delta

