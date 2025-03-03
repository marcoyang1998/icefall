#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"

for avg in $(seq 10 -2 4); do
    for chunk in 8 32 64; do
        for left in 256; do
            python zipformer_audio_encoder/decode.py \
                --iter 104000 \
                --avg $avg \
                --use-averaged-model 1 \
                --causal 1 \
                --chunk-size $chunk \
                --left-context-frames $left \
                --bpe-model data/lang_bbpe_2000/bbpe.model \
                --num-encoder-layers 2,2,4,5,4,2 \
                --feedforward-dim 512,1024,2048,3072,2048,1024 \
                --encoder-dim 192,384,768,1024,768,384 \
                --encoder-unmasked-dim 192,256,320,512,320,256 \
                --manifest-dir data/fbank_mtl \
                --on-the-fly-feats 1 \
                --exp-dir zipformer_audio_encoder_finetune/exp-xlarge-finetune-mtl-full-en-zh-lr-0.02-causal-1-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-use-mls-1-extra-zh-1-extra-en-1-at-scale-1.0-from-xlarge-lr-0.04-baoxiang-data-shar-500k \
                --decoding-method greedy_search \
                --max-duration 500
        done
    done
done

# for iter in 104000; do
#     for avg in $(seq 10 -2 2); do
#         for chunk in 8 32 64; do
#             for left in 256; do
#                 python zipformer_audio_encoder/decode_byte.py \
#                     --iter $iter \
#                     --avg $avg \
#                     --use-averaged-model 1 \
#                     --causal 1 \
#                     --chunk-size $chunk \
#                     --left-context-frames $left \
#                     --test-aishell 0 \
#                     --test-libri 0 \
#                     --test-wenet 1 \
#                     --num-encoder-layers 2,2,4,5,4,2 \
#                     --feedforward-dim 512,1024,2048,3072,2048,1024 \
#                     --encoder-dim 192,384,768,1024,768,384 \
#                     --encoder-unmasked-dim 192,256,320,512,320,256 \
#                     --manifest-dir data/fbank_mtl \
#                     --bpe-model data/lang_bbpe_2000/bbpe.model \
#                     --on-the-fly-feats 1 \
#                     --exp-dir zipformer_audio_encoder_finetune/exp-xlarge-finetune-mtl-full-en-zh-lr-0.02-causal-1-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-use-mls-1-extra-zh-1-extra-en-1-at-scale-1.0-from-xlarge-lr-0.04-baoxiang-data-shar-500k \
#                     --decoding-method greedy_search \
#                     --blank-penalty 0.0 \
#                     --max-duration 500
#             done
#         done
#     done
# done