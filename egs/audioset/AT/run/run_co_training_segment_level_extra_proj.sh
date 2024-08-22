#!/usr/bin/env bash

. /star-xy/miniconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate k2_cuda11

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="6,7"

echo $CUDA_VISIBLE_DEVICES

enable_spec_aug=0
use_spec_aug=1
use_time_warp=0
use_time_mask=1

subset=full
segment_length=10
output_downsampling_factor=2
co_training_scale=0.7
segment_level_at_loss_scale=0.1
segment_level_co_training_loss_scale=0.25

num_frame_masks=1
max_frames_mask_fraction=0.2
frames_mask_size=192

python zipformer_co_training/train_mtl_segment_level_extra_proj.py \
    --world-size 1 \
    --num-epochs 40 \
    --inf-check 0 \
    --exp-dir zipformer_co_training/exp_at_as_${subset}_co_training_extra_proj_segment_level_segment_length_${segment_length}_scale${co_training_scale}_segment_scale${segment_level_co_training_loss_scale}_aux_scale${segment_level_at_loss_scale}_matched_specaug_debug \
    --start-epoch 1 \
    --use-fp16 1 \
    --num-events 527 \
    --manifest-dir data/fbank \
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
    --num-frame-masks $num_frame_masks \
    --frames-mask-size $frames_mask_size \
    --max-frames-mask-fraction $max_frames_mask_fraction \
    --max-duration 500 \
    --enable-musan True \
    --master-port 13454