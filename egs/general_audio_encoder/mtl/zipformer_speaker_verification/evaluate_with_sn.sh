#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH



exp_dir=exp-316M-uniform-v2-out-ds-1-zipformer-weighted-sum-1-use-aam-1-margin-0.2-scale-30vox1-trunc-0-lr-0.025-musan-1-p-0.6-rir-0-p-0.5-sp-1-freeze-3-epochs-lr-scale-0.1-md-500

use_weighted_sum=1
use_aam=1
output_ds=1
post_output_ds=1

# for avg in $(seq $(( epoch - 3 )) -1 $(( epoch - 6 ))); do
for epoch in 4; do
    for avg in 1 2; do
        python zipformer_speaker_verification/score_normalization.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 0 \
            --manifest-dir data/fbank_voxceleb \
            --exp-dir zipformer_speaker_verification/$exp_dir \
            --downsampling-factor 1,2,4,8,4,2,1 \
            --num-encoder-layers 1,2,2,3,1,1,1 \
            --feedforward-dim 3072,3072,3072,3072,3072,3072,3072 \
            --encoder-dim 1024,1024,1024,1024,1024,1024,1024 \
            --encoder-unmasked-dim 512,512,512,512,512,512,512 \
            --cnn-module-kernel 31,31,15,15,15,31,31 \
            --num-heads 8,8,8,8,8,8,8 \
            --use-aam $use_aam \
            --use-weighted-sum $use_weighted_sum \
            --output-downsampling-factor $output_ds \
            --post-encoder-downsampling-factor $post_output_ds \
            --on-the-fly-feats 1 \
            --max-duration 400
    done
done