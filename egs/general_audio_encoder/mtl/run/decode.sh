#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"

for epoch in 25; do
    for avg in 5; do
        for chunk in 8 32 64; do
            for left in 256; do
                python zipformer_audio_encoder/decode.py \
                    --epoch $epoch \
                    --avg $avg \
                    --use-averaged-model 1 \
                    --causal 1 \
                    --chunk-size $chunk \
                    --left-context-frames $left \
                    --num-encoder-layers 2,2,4,5,4,2 \
                    --feedforward-dim 512,1024,2048,3072,2048,1024 \
                    --encoder-dim 192,384,768,1024,768,384 \
                    --encoder-unmasked-dim 192,256,320,512,320,256 \
                    --manifest-dir data/fbank_mtl \
                    --bpe-model data/lang_bpe_500/bpe.model \
                    --on-the-fly-feats 1 \
                    --exp-dir zipformer_audio_encoder_finetune/exp-finetune-ls-100h-lr-0.045-causal-1-freeze-encoder-0-freeze-2000-step-encoder-lr-scale-0.05-from-xlarge-extra-baoxiang-data-350k \
                    --decoding-method greedy_search \
                    --max-duration 1000
            done
        done
    done
done

# for epoch in 30; do
#     for avg in $(seq 15 -1 10); do
#         for chunk in 8 32 64; do
#             for left in 256; do
#                 python zipformer_audio_encoder/decode.py \
#                     --epoch $epoch \
#                     --avg $avg \
#                     --use-averaged-model 1 \
#                     --causal 1 \
#                     --chunk-size $chunk \
#                     --left-context-frames $left \
#                     --num-encoder-layers 2,2,4,5,4,2 \
#                     --feedforward-dim 512,768,1536,2048,1536,768 \
#                     --encoder-dim 192,256,512,768,512,256 \
#                     --encoder-unmasked-dim 192,192,256,320,256,192 \
#                     --manifest-dir data/fbank_mtl \
#                     --on-the-fly-feats 1 \
#                     --exp-dir zipformer_audio_encoder_finetune/exp-finetune-ls-100h-lr-0.045-causal-1-freeze-encoder-0-freeze-2000-step-encoder-lr-scale-0.05-from-large-shar-en-mvq-audio-mvq-0.1-lr-0.04-200k \
#                     --decoding-method greedy_search \
#                     --max-duration 500
#             done
#         done
#     done
# done