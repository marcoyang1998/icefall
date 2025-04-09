#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH


python zipformer_audio_encoder/export_mtl.py \
    --iter 196000 \
    --avg 8 \
    --exp-dir zipformer_audio_encoder_finetune/exp-xlarge-finetune-mtl-full-en-zh-lr-0.02-causal-1-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-use-mls-1-use-weread-1-extra-zh-1-extra-en-1-at-scale-2.0-from-xlarge-lr-0.04-from-v1.0-500k-full-data-fix-lh-ft-with-musan-rir-fix-lh \
    --use-averaged-model 1 \
    --causal 1 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --chunk-size 8,32,64,-1 \
    --left-context-frames 64,128,-1 \
    --use-s3-client True \
    --bpe-model data/lang_bbpe_2000/bbpe.model \
    --context-size 2
exit
