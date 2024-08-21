#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/xy/mnt/yangxiaoyu/workspace/icefall_multi_KD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="3"


for iter in $(seq 516000 -4000 492000); do
    for avg in 3 4 5; do
        python multi_KD/inference_speaker.py \
            --iter $iter \
            --avg $avg --use-averaged-model 0 \
            --causal 1 \
            --chunk-size 32 \
            --left-context-frames 256 \
            --exp-dir multi_KD/exp_causal1_delta6KD_LS1_5fold+wenetspech0_0fold+as_unbalanced1+vox_1_vox2_base_lr_0.045_use_beats_1_scale_1.0_use_ecapa_1_layer_2_scale_10.0_1_scale_1.0_specaug0_musan0_with_task_ID_stop_early1_share_asr1_md1500_amp_bf16 \
            --num-workers 2 \
            --max-duration 400 \
            --return-audio 0 \
            --use-beats 1 \
            --use-whisper 1 \
            --whisper-dim 1280 \
            --manifest-dir data/fbank_multi_KD \
            --use-ecapa 1 --speaker-input-idx 2 \
            --trained-with-distillation 1 \
            --on-the-fly-feats 0
    done
done