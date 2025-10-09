#!/usr/bin/env bash

model_version=600m_uniform_out_ds1
causal=1
left_context_frames=256
chunk_size=8

python zipformer_audio_encoder_clean/inference_600m_streaming_forward.py \
    --model-version $model_version \
    --ckpt-path zipformer_audio_encoder/exp-600M-uni-v2-out-ds-1-causal-1-lh-large-giga-xl-voxpopuli-1-yodas-0-lr-batches-7500-lr-hours-75000-w2v2-mask-p-0.5-l-10-cha-mask-p-0.25-l-15-batch-mix-p-0.3-min-snr--5-p-noise-0.7-min-snr-5-wavlm-cb-16-shar-md-400/iter-300000-avg-4.pt \
    --causal $causal \
    --left-context-frames $left_context_frames \
    --chunk-size $chunk_size \
    --audio download/LibriSpeech/dev-clean/5338/24640/5338-24640-0000.flac