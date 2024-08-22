#!/usr/bin/env bash

source ~/softwares/pyenvs/k2_cuda11/k2_cuda11/bin/activate

source new_env.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo $CUDA_VISIBLE_DEVICES

enable_spec_aug=0
use_spec_aug=1
use_time_warp=1
use_time_mask=1

subset=full
segment_length=10
output_downsampling_factor=2
co_training_scale=0.7
segment_level_at_loss_scale=0.1
segment_level_co_training_loss_scale=0.1

weightd_sampler=1
bucketing_sampler=0
lr_epochs=30
md=1000

num_frame_masks=1
max_frames_mask_fraction=0.15
frames_mask_size=192
features_mask_size=27
feature_dim=128
use_beats=0
use_KD=1

python zipformer_co_training/train_mtl_segment_level_extra_proj_KD.py \
    --world-size 4 \
    --num-epochs 150 \
    --exp-dir zipformer_co_training/exp_KD_CED_base_at_as_${subset}_co_training_extra_proj_segment_level_segment_length_${segment_length}_scale${co_training_scale}_segment_scale${segment_level_co_training_loss_scale}_aux_scale${segment_level_at_loss_scale}_matched_specaug_num_masks${num_frame_masks}_mask_span${frames_mask_size}_feature_mask${features_mask_size}_lr_epochs${lr_epochs}_weighted${weightd_sampler}_md${md}-128D \
    --start-epoch 1 \
    --use-fp16 1 \
    --num-events 527 \
    --manifest-dir data/fbank_as_ced_mAP50 \
    --output-downsampling-factor $output_downsampling_factor \
    --co-training-loss-scale $co_training_scale \
    --segment-level-at-loss-scale $segment_level_at_loss_scale \
    --segment-level-co-training-loss-scale $segment_level_co_training_loss_scale \
    --segment-length $segment_length \
    --enable-spec-aug $enable_spec_aug \
    --use-spec-aug $use_spec_aug \
    --use-time-warp $use_time_warp \
    --use-time-mask $use_time_mask \
    --audioset-subset $subset \
    --lr-epochs $lr_epochs \
    --weighted-sampler $weightd_sampler \
    --bucketing-sampler $bucketing_sampler \
    --feature-dim $feature_dim \
    --features-mask-size $features_mask_size \
    --frames-mask-size $frames_mask_size \
    --max-frames-mask-fraction $max_frames_mask_fraction \
    --max-duration $md \
    --use-KD $use_KD \
    --use-beats $use_beats \
    --enable-musan True \
    --master-port 13444