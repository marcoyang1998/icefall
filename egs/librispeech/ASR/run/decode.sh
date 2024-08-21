#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/xy/mnt/yangxiaoyu/workspace/icefall_multi_KD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"


exp_dir=exp_finetune_asr_libri1x6_do_AT1_unbalanced_KD_scale2.0_do_SV0_only_vox2_scale10.0_freeze_12000steps_encoder_lr_scale0.2_freeze_3layers_ecapa_lr_scale0.2_init_3_tasks_delta6_pretrain_avg_musan0_sync_task_md1000_amp_bf16

for m in greedy_search; do
    for epoch in $(seq 18 -1 15); do
        for avg in $(seq 8 -1 3); do
            python multi_KD/decode.py \
                --epoch $epoch \
                --avg $avg \
                --use-averaged-model 1 \
                --beam-size 4 \
                --exp-dir multi_KD/$exp_dir \
                --use-librispeech 1 \
                --use-bpe 1 \
                --use-wenetspeech 0 \
                --manifest-dir data/fbank \
                --max-duration 500 \
                --beats-label 0 \
                --causal 1 \
                --chunk-size 32 \
                --left-context-frames 256 \
                --do-sv 0 --do-audio-tagging 0 \
                --use-encoder-projection  1 \
                --encoder-projection-dim 2560 \
                --decoding-method $m
        done
    done
done