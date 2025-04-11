#!/usr/bin/env bash

export PYTHONPATH=./../../..:$PYTHONPATH

model_name=dasheng

manifest_dir=data/manifests/${model_name}
embedding_dir=data/embeddings/${model_name}
mkdir -p $manifest_dir
mkdir -p $embedding_dir

embedding_layer=-1
model_version=large

export CUDA_VISIBLE_DEVICES="0,1"
for subset in eval; do
    python dasheng/collect_embeddings.py \
        --num-jobs 2 \
        --model-version $model_version \
        --input-manifest data/fbank_audioset/audioset_cuts_${subset}.jsonl.gz \
        --manifest-name audioset-${subset} \
        --target-manifest-file ${manifest_dir}/${model_name}-layer-${embedding_layer}-audioset-${subset}.jsonl.gz \
        --embedding-dir $embedding_dir \
        --embedding-layer $embedding_layer \
        --max-duration 200
done
