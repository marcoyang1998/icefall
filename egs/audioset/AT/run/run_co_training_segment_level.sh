#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate multi_KD

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
export LD_LIBRARY_PATH=/star-xy/softwares/anaconda3/envs/k2_latest/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="4,5,6,7"

echo $CUDA_VISIBLE_DEVICES

enable_spec_aug=0
use_spec_aug=1
use_time_warp=1
use_time_mask=1

subset=full
segment_length=10
output_downsampling_factor=2
co_training_scale=0.8
segment_level_loss_scale=0.25


python zipformer_co_training/train_mtl_segment_level.py \
    --world-size 1 \
    --num-epochs 40 \
    --inf-check 0 \
    --exp-dir zipformer_co_training/exp_at_as_${subset}_co_training_segment_level_segment_length_${segment_length}_scale${co_training_scale}_segment_scale${segment_level_loss_scale}_matched_specaug_debug \
    --start-epoch 1 \
    --use-fp16 1 \
    --num-events 527 \
    --manifest-dir data/fbank \
    --output-downsampling-factor $output_downsampling_factor \
    --co-training-loss-scale $co_training_scale \
    --segment-level-loss-scale $segment_level_loss_scale \
    --segment-length $segment_length \
    --enable-spec-aug $enable_spec_aug \
    --use-spec-aug $use_spec_aug \
    --use-time-warp $use_time_warp \
    --use-time-mask $use_time_mask \
    --audioset-subset $subset \
    --max-duration 500 \
    --enable-musan True \
    --master-port 13454