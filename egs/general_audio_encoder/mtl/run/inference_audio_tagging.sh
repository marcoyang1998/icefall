#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=~/xiaoyu/workspace/lhotse:$PYTHONPATH

do_mvq=0
num_codebooks=0
delta=3
frame_rate_ratio=2

exp_dir=exp-finetune-AT-as-balanced-lr-0.045-musan-0-mixup-0.5-freeze-encoder-0-freeze-2000-step-encoder-lr-scale-0.05-from-small-decoder-mae-scale-0.2-w2v2-mask-p-0.5-channel-0.25-dasheng-as-mvq-cb8-with-musan-300k

for epoch in $(seq 14 1 26); do
    for avg in $(seq $(( epoch - 8)) 1 $(( epoch - 4))); do
        python zipformer_audio_encoder/inference_audio_tagging.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --at-KD 0 \
            --do-mvq $do_mvq \
            --num-codebooks $num_codebooks --teacher-frame-ratio $frame_rate_ratio \
            --exp-dir zipformer_audio_encoder_finetune/$exp_dir \
            --num-encoder-layers 2,2,3,4,3,2 \
            --feedforward-dim 512,768,1024,1536,1024,768 \
            --encoder-dim 192,256,448,768,448,192 \
            --encoder-unmasked-dim 192,192,256,256,256,192 \
            --manifest-dir data_s3/vq_dasheng_large_cb_16 \
            --on-the-fly-feats 1 \
            --causal 0 \
            --max-duration 1000
    done
done