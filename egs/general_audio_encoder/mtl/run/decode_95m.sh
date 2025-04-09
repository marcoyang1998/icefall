#!/usr/bin/env bash

source ~/anaconda3/bin/activate && conda activate encoder
source /mnt/cache/share_data/housiyuan/softwares/activate-cuda-11.8.sh

# export PYTHONPATH=/mnt/cache/share_data/housiyuan/icefall_audio_encoder:$PYTHONPATH
export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH


output_ds=2
post_output_ds=1

for epoch in $(seq 200 4 220); do
    for avg in $(seq 210 -10 160); do
        python zipformer_audio_encoder/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --num-encoder-layers 2,2,3,4,3,2 \
            --feedforward-dim 512,768,1024,1536,1024,768 \
            --encoder-dim 192,256,448,768,448,192 \
            --encoder-unmasked-dim 192,192,256,256,256,192 \
            --manifest-dir data_s3/fbank_librispeech \
            --output-downsampling-factor $output_ds \
            --post-encoder-downsampling-factor $post_output_ds \
            --on-the-fly-feats 1 \
            --exp-dir zipformer_audio_encoder_finetune/exp-finetune-95M-ls-100h-lr-0.02-causal-0-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-from-whisper-mvq-cb16--v2-musan-rir-224k \
            --decoding-method modified_beam_search \
            --beam-size 8 \
            --num-workers 20 \
            --max-duration 500
    done
done