#!/usr/bin/env bash

source ~/anaconda3/bin/activate && conda activate encoder
source /mnt/cache/share_data/housiyuan/softwares/activate-cuda-11.8.sh

export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH

# for avg in $(seq 15 -1 11); do
#     for chunk in 8 32 64; do
#         for left in 256; do
#             python zipformer_audio_encoder/decode_byte.py \
#                 --epoch 30 \
#                 --avg $avg \
#                 --use-averaged-model 1 \
#                 --causal 1 \
#                 --chunk-size $chunk \
#                 --left-context-frames $left \
#                 --test-aishell 0 \
#                 --test-libri 0 \
#                 --test-wenet 1 \
#                 --bpe-model data/lang_bbpe_2000/bbpe.model \
#                 --num-encoder-layers 2,2,4,5,4,2 \
#                 --feedforward-dim 512,1024,2048,3072,2048,1024 \
#                 --encoder-dim 192,384,768,1024,768,384 \
#                 --encoder-unmasked-dim 192,256,320,512,320,256 \
#                 --manifest-dir data_s3/fbank_wenetspeech \
#                 --on-the-fly-feats 1 \
#                 --exp-dir zipformer_audio_encoder_finetune/exp-xlarge-finetune-mtl-full-en-zh-lr-0.02-causal-1-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-use-mls-1-use-weread-1-extra-zh-1-extra-en-1-at-scale-2.0-from-xlarge-lr-0.04-from-v1.0-500k-full-data-fix-lh-ft-with-musan-rir-fix-lh \
#                 --decoding-method greedy_search \
#                 --max-duration 1000
#         done
#     done
# done

# exit

for iter in 24000; do
    for avg in 1; do
        for chunk in 8 32; do
            for left in 256; do
                python zipformer_audio_encoder/decode_byte.py \
                    --iter $iter \
                    --avg $avg \
                    --use-averaged-model 1 \
                    --causal 1 \
                    --chunk-size $chunk \
                    --left-context-frames $left \
                    --test-aishell 0 \
                    --test-libri 0 \
                    --test-wenet 1 \
                    --num-encoder-layers 2,2,4,5,4,2 \
                    --feedforward-dim 512,1024,2048,3072,2048,1024 \
                    --encoder-dim 192,384,768,1024,768,384 \
                    --encoder-unmasked-dim 192,256,320,512,320,256 \
                    --joiner-dim 768 \
                    --decoder-dim 768 \
                    --manifest-dir data_s3/fbank_wenetspeech \
                    --bpe-model data/lang_bbpe_2000/bbpe.model \
                    --on-the-fly-feats 1 \
                    --exp-dir zipformer_audio_encoder_finetune/exp-xlarge-finetune-mtl-full-en-zh-lr-0.02-causal-1-freeze-encoder-0-freeze--1-step-encoder-lr-scale-0.1-use-mls-1-use-weread-1-extra-zh-1-extra-en-1-at-scale-2.0-from-xlarge-lr-0.04-from-v1.0-500k-full-data-fix-lh-ft-with-musan-rir-fix-lh-bigger-decoder \
                    --decoding-method greedy_search \
                    --blank-penalty 0.0 \
                    --num-workers 20 \
                    --max-duration 1000
            done
        done
    done
done