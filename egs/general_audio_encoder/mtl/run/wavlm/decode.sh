#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

wavlm_version=base
encoder_dim=768

export CUDA_VISIBLE_DEVICES="0"


for m in greedy_search modified_beam_search; do
    for epoch in $(seq 15 1 24); do
        for avg in $(seq $(( epoch - 12 )) 1 $(( epoch - 7)) ); do    
            python wavlm_pretrain/decode.py \
                --epoch $epoch \
                --avg $avg \
                --use-averaged-model 1 \
                --wavlm-version $wavlm_version \
                --encoder-dim $encoder_dim \
                --manifest-dir data/fbank_librispeech \
                --exp-dir wavlm_pretrain/exp-finetune-wavlm-base-ls-100h-opt-adamw-lr-5e-4-scheduler--warmup-10000-layerdrop-0.05-freeze-5000-step-from-hubert-large-layer-21-mvq-cb16-mask-prob-0.65-300k-md300 \
                --decoding-method $m \
                --max-duration 300
        done
    done
done