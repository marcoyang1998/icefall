#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH


python zipformer_inference/inference_model.py \
    --ckpt-path zipformer_audio_encoder/exp-316M-uniform-v2-out-ds-1-zipformer-lh-large-giga-xl-emo-0-voxpopuli-1-lr-batches-7500-lr-hours-75000-w2v2-mask-p-0.5-l-10-cha-mask-p-0.25-l-15-batch-mix-p-0.3-min-snr--5-p-noise-0.7-min-snr-5-wavlm-large-layer-21-libri-mvq-cb-16-shar-md-400-16-gpus/iter-400000-avg-4.pt \
    --output-downsampling-factor 1 \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,2,3,1,1,1 \
    --feedforward-dim 3072,3072,3072,3072,3072,3072,3072 \
    --encoder-dim 1024,1024,1024,1024,1024,1024,1024 \
    --encoder-unmasked-dim 512,512,512,512,512,512,512 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --audio /cpfs02/shared/speechllm/data/LibriSpeech/LibriSpeech/dev-clean/5338/24640/5338-24640-0000.flac