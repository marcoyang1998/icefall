#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="1"

output_ds=2
post_output_ds=1

for epoch in 30 40; do
    for avg in 15 20; do
        python zipformer_audio_encoder/decode_byte.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 0 \
            --test-aishell 0 \
            --test-libri 0 \
            --test-wenet 1 \
            --num-encoder-layers 2,2,3,4,3,2 \
            --feedforward-dim 512,768,1024,1536,1024,768 \
            --encoder-dim 192,256,448,768,448,192 \
            --encoder-unmasked-dim 192,192,256,256,256,192 \
            --manifest-dir data/fbank_mtl \
            --bpe-model data/lang_bbpe_2000/bbpe.model \
            --on-the-fly-feats 1 \
            --exp-dir zipformer_audio_encoder_finetune/exp-finetune-95M-wenet-S-lr-0.02-causal-0-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-from-firered-en-zh-mvq-cb16-no-musan-mask-ratio-1.0-200k \
            --decoding-method greedy_search \
            --blank-penalty 0.0 \
            --max-duration 500
    done
done