#!/usr/bin/env bash

source ~/softwares/pyenvs/k2_cuda11/k2_cuda11/bin/activate

source new_env.sh
export CUDA_VISIBLE_DEVICES="4,5,6,7"

echo $CUDA_VISIBLE_DEVICES

subset=full

enable_spec_aug=1
enable_musan=1
frames_mask_size=192
features_mask_size=27

weightd_sampler=1
bucketing_sampler=0
lr_epochs=15
md=1000
feature_dim=128

use_beats=0
use_KD=0


python zipformer/train_bf16.py \
    --world-size 4 \
    --num-epochs 100 \
    --exp-dir zipformer/exp_at_${subset}_lr_epochs_${lr_epochs}_specaug${enable_spec_aug}_frame${frames_mask_size}_feature${features_mask_size}_musan${enable_musan}_weighted${weightd_sampler}_md${md}_bf16 \
    --start-epoch 1 \
    --use-fp16 1 --use-bf16 1 --full-bf16 1 \
    --num-events 527 \
    --audioset-subset $subset \
    --feature-dim $feature_dim \
    --manifest-dir data/fbank_as_ced_mAP50 \
    --enable-spec-aug $enable_spec_aug \
    --enable-musan $enable_musan \
    --frames-mask-size $frames_mask_size \
    --features-mask-size $features_mask_size \
    --weighted-sampler $weightd_sampler \
    --bucketing-sampler $bucketing_sampler \
    --lr-epochs $lr_epochs \
    --max-duration $md \
    --use-KD $use_KD \
    --use-beats $use_beats \
    --master-port 14394

