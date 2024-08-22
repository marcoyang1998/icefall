#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate multi_KD

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="2,3"

echo $CUDA_VISIBLE_DEVICES

subset=balanced
output_downsampling_factor=4

python zipformer/train.py \
    --world-size 2 \
    --num-epochs 50 \
    --exp-dir zipformer/exp_small_at_as_${subset}_output_downsample${output_downsample_factor} \
    --start-epoch 1 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --output-downsampling-factor $output_downsampling_factor \
    --use-fp16 1 \
    --num-events 527 \
    --audioset-subset $subset \
    --max-duration 1200 \
    --enable-musan True \
    --master-port 13452