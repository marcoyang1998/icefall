#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/xy/mnt/yangxiaoyu/workspace/icefall_multi_KD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="5"

# iter=492000
iter=468000
avg=5
use_averaged_model=0

python multi_KD/export.py \
    --iter $iter \
    --use-averaged-model $use_averaged_model \
    --avg $avg \
    --use-beats 1 \
    --use-ecapa 1 \
    --causal 1 \
    --speaker-input-idx 2 \
    --whisper-dim 1280 \
    --exp-dir multi_KD/exp_causal1_delta6KD_LS1_5fold+wenetspech0_0fold+as_unbalanced1+vox_1_vox2_base_lr_0.045_use_beats_1_scale_1.0_use_ecapa_1_layer_2_scale_10.0_1_scale_1.0_specaug0_musan0_with_task_ID_stop_early1_share_asr1_md1500_amp_bf16