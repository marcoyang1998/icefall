#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate multi_KD
# source /star-fj/fangjun/software/activate-cuda-12.1.sh
# nvcc -V

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
export LD_LIBRARY_PATH=/star-xy/softwares/anaconda3/envs/k2_latest/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo $CUDA_VISIBLE_DEVICES

enable_spec_aug=0
use_spec_aug=1
use_time_warp=1
use_time_mask=1

subset=full
frame_level=1
output_downsampling_factor=2
co_training_scale=0.7
frame_level_at_loss_scale=0.1
frame_level_co_training_scale=0.2


python zipformer_co_training/train_mtl_mask_postions_extra_proj.py \
    --world-size 4 \
    --num-epochs 40 \
    --inf-check 0 \
    --exp-dir zipformer_co_training/exp_at_as_${subset}_co_training_extra_proj_bi_level_scale${co_training_scale}_frame_scale${frame_level_co_training_scale}_downsample${output_downsampling_factor}_aux_scale${frame_level_at_loss_scale}_mask_position_matched_specaug \
    --start-epoch 1 \
    --use-fp16 1 \
    --num-events 527 \
    --manifest-dir data/fbank \
    --output-downsampling-factor $output_downsampling_factor \
    --co-training-loss-scale $co_training_scale \
    --frame-level-co-training $frame_level \
    --frame-level-co-training-scale $frame_level_co_training_scale \
    --frame-level-at-loss-scale $frame_level_at_loss_scale \
    --enable-spec-aug $enable_spec_aug \
    --use-spec-aug $use_spec_aug \
    --use-time-warp $use_time_warp \
    --use-time-mask $use_time_mask \
    --audioset-subset $subset \
    --max-duration 500 \
    --enable-musan True \
    --master-port 13450