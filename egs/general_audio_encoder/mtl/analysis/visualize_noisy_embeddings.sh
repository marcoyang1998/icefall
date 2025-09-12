#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH

# python analysis/analyze_noisy_embeddings.py \
#     --input-manifest ./data/manifests/vox1_cuts_test_noisy.jsonl.gz \
#     --speaker-group 0 \
#     --embedding-layer 21 \
#     --suffix wavlm-large-layer-21

num_cb=16

python analysis/analyze_noisy_embeddings_with_mvq.py \
    --input-manifest data/manifests/librispeech_dev_clean_noisy.jsonl.gz \
    --speaker-group 0 \
    --embedding-layer 21 \
    --quantizer-path data/quantizer/wavlm-large-layer-21-normalize-1-libri-cb-${num_cb}.pt \
    --num-codebooks $num_cb \
    --suffix wavlm-large-layer-21-ls-dev-clean