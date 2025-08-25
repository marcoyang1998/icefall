#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

num_codebooks=0
delta=0
output_ds=2
teacher_frame_ratio=2

python zipformer_audio_encoder/export_asr.py \
    --epoch 16 \
    --avg 15 \
    --exp-dir zipformer_audio_encoder_finetune/exp-finetune-316M-full-libri-lr-0.02-causal-0-freeze-encoder-0-encoder-lr-scale-0.1-warp-80-musan-1-from-hubert-mvq-cb16-lh-giga-emo-pretrain-400k \
    --output-downsampling-factor $output_ds \
    --use-averaged-model 1 \
    --causal 0 \
    --num-encoder-layers 2,2,4,6,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --chunk-size 8,16,32,64,-1 \
    --left-context-frames 64,128,-1

