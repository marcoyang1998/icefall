#!/usr/bin/env bash

dataset=wenetspeech
mkdir -p data/fbank_${dataset}
for subset in DEV TEST_NET TEST_MEETING S M L; do
    if [ ! -f data/fbank_${dataset}/${dataset}_cuts_${subset}.jsonl.gz ]; then
        python join_manifest.py \
            --recording-manifest /fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall/egs/wenetspeech/ASR/data/manifests/${dataset}_recordings_${subset}.jsonl.gz \
            --supervision-manifest /fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall/egs/wenetspeech/ASR/data/manifests/${dataset}_supervisions_${subset}.jsonl.gz \
            --output-manifest data/fbank_${dataset}/${dataset}_cuts_${subset}.jsonl.gz
    fi
done