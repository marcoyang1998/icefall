#!/usr/bin/env bash

source ~/anaconda3/bin/activate && conda activate encoder
source /mnt/cache/share_data/housiyuan/softwares/activate-cuda-11.8.sh

export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH

do_mvq=0
num_codebooks=0
delta=3
frame_rate_ratio=2

for iter in 196000; do
    for avg in 8; do
        for chunk in 8 32; do
            python zipformer_audio_encoder/inference_audio_tagging.py \
                --iter $iter \
                --avg $avg \
                --use-averaged-model 1 \
                --at-KD 0 \
                --do-mvq $do_mvq \
                --num-codebooks $num_codebooks --teacher-frame-ratio $frame_rate_ratio \
                --exp-dir zipformer_audio_encoder_finetune/exp-xlarge-finetune-mtl-full-en-zh-lr-0.02-causal-1-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-use-mls-1-use-weread-1-extra-zh-1-extra-en-1-at-scale-2.0-from-xlarge-lr-0.04-from-v1.0-500k-full-data-fix-lh-ft-with-musan-rir-fix-lh \
                --num-encoder-layers 2,2,4,5,4,2 \
                --feedforward-dim 512,1024,2048,3072,2048,1024 \
                --encoder-dim 192,384,768,1024,768,384 \
                --encoder-unmasked-dim 192,256,320,512,320,256 \
                --use-shar 1 --shar-dir data-shar/data-shar-whisper-zh-en-cb16-v2 \
                --manifest-dir data/fbank_as_ced_mAP50 \
                --on-the-fly-feats 1 \
                --causal 1 \
                --chunk-size $chunk \
                --num-workers 20 \
                --left-context 256 \
                --max-duration 1000
        done
    done
done