#!/usr/bin/env bash

source ~/anaconda3/bin/activate && conda activate encoder
source /mnt/cache/share_data/housiyuan/softwares/activate-cuda-11.8.sh

export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH


output_ds=2
post_output_ds=1

for epoch in 30; do
    for avg in 15 20; do
        python zipformer_audio_encoder/decode_byte.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 0 \
            --test-aishell 0 \
            --test-libri 0 \
            --test-wenet 1 \
            --num-encoder-layers 2,2,3,4,3,2 \
            --feedforward-dim 512,768,1024,1536,1024,768 \
            --encoder-dim 192,256,448,768,448,192 \
            --encoder-unmasked-dim 192,192,256,256,256,192 \
            --manifest-dir data/fbank_mtl \
            --bpe-model data/lang_bbpe_2000/bbpe.model \
            --on-the-fly-feats 1 \
            --exp-dir zipformer_audio_encoder_finetune/exp-finetune-wenetspeech-S-lr-0.045-causal-1-freeze-encoder-0-freeze-2000-step-encoder-lr-scale-0.05-from-xlarge-all-data-with-musan-rir-348k-fix \
            --decoding-method greedy_search \
            --blank-penalty 0.0 \
            --max-duration 800
    done
done