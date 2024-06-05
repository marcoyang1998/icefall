#!/usr/bin/env bash

. /star-xy/miniconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate k2_cuda11

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_ssl:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo $CUDA_VISIBLE_DEVICES

subset=full
mask_prob=0.65
enable_spec_aug=0
enable_musan=0
noise_scale=0.1

python zipformer_pretraining/train_mask_fbank_only_masked.py \
    --world-size 4 \
    --audioset-subset $subset \
    --num-epochs 30 \
    --start-epoch 1 \
    --use-fp16 1 \
    --enable-spec-aug $enable_spec_aug \
    --enable-musan $enable_musan \
    --noise-scale $noise_scale \
    --exp-dir zipformer_pretraining/exp_${subset}_mask_fbank_only_masked_frames_mask_prob${mask_prob}_specaug${enable_spec_aug}_musan${enable_musan}_noise${noise_scale}_fix_rand \
    --mask-prob $mask_prob \
    --max-duration 1000 \
    --master-port 13840
