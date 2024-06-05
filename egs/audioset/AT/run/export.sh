#!/usr/bin/env bash

. /star-xy/miniconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate k2_cuda11

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_ssl:$PYTHONPATH

python zipformer_pretraining/export_mask_fbank.py \
    --exp-dir zipformer_pretraining/exp_full_mask_fbank_only_masked_frames_mask_prob0.65_specaug0_musan0_noise0.1 \
    --epoch 30 \
    --avg 5 \
    --use-averaged-model 1 \
    --jit 0
