#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate multi_KD
# source /star-fj/fangjun/software/activate-cuda-12.1.sh
# nvcc -V

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
export LD_LIBRARY_PATH=/star-xy/softwares/anaconda3/envs/k2_latest/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="4,5"

echo $CUDA_VISIBLE_DEVICES

subset=balanced
co_training_scale=0.7

python zipformer_co_training/train.py \
    --world-size 2 \
    --num-epochs 50 \
    --exp-dir zipformer_co_training/exp_at_as_${subset}_co_training_scale${co_training_scale} \
    --start-epoch 1 \
    --use-fp16 1 \
    --num-events 527 \
    --manifest-dir data/fbank \
    --co-training-loss-scale $co_training_scale \
    --audioset-subset $subset \
    --max-duration 500 \
    --enable-musan True \
    --master-port 13454