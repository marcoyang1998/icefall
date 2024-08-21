#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/xy/mnt/yangxiaoyu/workspace/icefall_multi_KD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="1"

for iter in $(seq 516000 -4000 496000); do
    for avg in 3 4 5; do
        for avg_model in 0; do
            python multi_KD/inference_audio_tagging.py \
                --iter $iter \
                --avg $avg --use-averaged-model $avg_model \
                --exp-dir multi_KD/exp_causal1_delta6KD_LS1_5fold+wenetspech0_0fold+as_unbalanced1+vox_1_vox2_base_lr_0.045_use_beats_1_scale_1.0_use_ecapa_1_layer_2_scale_10.0_1_scale_1.0_specaug0_musan0_with_task_ID_stop_early1_share_asr1_md1500_amp_bf16 \
                --causal 1 \
                --chunk-size 32 \
                --left-context-frames 256 \
                --manifest-dir data/fbank_multi_KD \
                --max-duration 300 \
                --return-audio 0 \
                --eval-subset eval \
                --use-beats 1 \
                --use-whisper 1 \
                --whisper-dim 1280 \
                --use-ecapa 0 --speaker-input-idx 2 \
                --use-encoder-projection 1 \
                --encoder-projection-dim 2560 \
                --trained-with-distillation 1 \
                --trained-with-multitask 0 \
                --on-the-fly-feats 0
        done
    done
done