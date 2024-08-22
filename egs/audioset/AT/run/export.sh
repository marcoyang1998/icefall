#!/usr/bin/env bash

source ~/softwares/pyenvs/k2_cuda11/k2_cuda11/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="4"

python zipformer/export.py \
    --exp-dir zipformer/exp_KD_CED_base_full_lr_epochs_15_specaug1_frame192_feature27_musan1_weighted1_md1000 \
    --epoch 150 \
    --avg 18 \
    --feature-dim 128 \
    --use-averaged-model 1 \
    --jit 0