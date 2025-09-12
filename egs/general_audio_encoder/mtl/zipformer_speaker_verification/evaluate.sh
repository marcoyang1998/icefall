#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH



exp_dir=exp-316M-uniform-v2-out-ds-1-zipformer-weighted-sum-1-adam-lr-0.001-lr-batches-7500-use-aam-1-margin-0.2-scale-30-vox2-trunc-0-musan-1-p-0.6-rir-1-p-0.5-sp-1-freeze-6-epochs-lr-scale-0.1-md-500

speaker_embed_dim=512
num_speakers=17982
use_weighted_sum=1
use_aam=1
output_ds=1
post_output_ds=1

# for avg in $(seq $(( epoch - 3 )) -1 $(( epoch - 6 ))); do
for iter in 88000 84000; do
    for avg in 1; do
        python zipformer_speaker_verification/compute_eer.py \
            --iter $iter \
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
            --num-speakers $num_speakers --speaker-embed-dim $speaker_embed_dim \
            --use-aam $use_aam \
            --use-weighted-sum $use_weighted_sum \
            --output-downsampling-factor $output_ds \
            --post-encoder-downsampling-factor $post_output_ds \
            --on-the-fly-feats 1 \
            --max-duration 200
    done
done