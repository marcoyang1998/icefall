#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES="0,1,2,3"

for part in eval; do
    python multi_KD/collect_ced_embeddings.py \
        --num-jobs 2 \
        --max-duration 200 \
        --input-manifest ./data/fbank_audioset/audioset_cuts_${part}.jsonl.gz \
        --output-manifest embeddings-audioset-${part} \
        --ced-ckpt audiotransformer_base_mAP_4999.pt \
        --model-id CED-base-mAP50 \
        --embedding-dir ./data/fbank_as_ced_mAP50 \
        --target-manifest-file ./data/fbank_as_ced_mAP50/audioset_cuts_${part}.jsonl.gz
done