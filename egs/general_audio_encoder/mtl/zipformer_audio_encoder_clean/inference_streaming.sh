#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

causal=1
left_context_frames=256
chunk_size=8


python zipformer_audio_encoder_clean/inference_streaming.py \
    --ckpt-path zipformer_audio_encoder/exp-600M-uni-v2-out-ds-1-causal-1-lh-large-giga-xl-voxpopuli-1-yodas-0-lr-batches-7500-lr-hours-75000-w2v2-mask-p-0.5-l-10-cha-mask-p-0.25-l-15-batch-mix-p-0.3-min-snr--5-p-noise-0.7-min-snr-5-wavlm-cb-16-shar-md-400/iter-300000-avg-4.pt \
    --output-downsampling-factor 1 \
    --causal $causal \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,3,4,1,1,1 \
    --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
    --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
    --encoder-unmasked-dim 768,768,768,768,768,768,768 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --left-context-frames $left_context_frames \
    --chunk-size $chunk_size \
    --audio download/LibriSpeech/dev-clean/5338/24640/5338-24640-0000.flac