#!/usr/bin/env bash

model_version=600m_uniform_out_ds1
causal=1
left_context_frames=128
chunk_size=8

python zipformer_audio_encoder_clean/inference_600m_streaming_forward.py \
    --model-version $model_version \
    --ckpt-path zipformer_audio_encoder_finetune/exp-finetune-uniform-600m-out-ds-1-causal-1-ls-100h-lr-0.045-freeze-2000-step-encoder-lr-scale-0.05-from-wavlm-mvq-lh-giga-vox-300k/epoch-30.pt \
    --causal $causal \
    --left-context-frames $left_context_frames \
    --chunk-size $chunk_size \
    --audio download/LibriSpeech/test-clean/1284/1180/1284-1180-0027.flac