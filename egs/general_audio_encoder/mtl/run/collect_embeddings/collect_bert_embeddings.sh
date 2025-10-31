#!/usr/bin/env bash

export PYTHONPATH=./../../..:$PYTHONPATH

# mount data
bash mount_brainllm_h.sh 

model_name=bert

manifest_dir=data/manifests/${model_name}
embedding_dir=data/embeddings/${model_name}
mkdir -p $manifest_dir
mkdir -p $embedding_dir

model_version=large
embedding_layer=-1

for subset in dev-clean dev-other; do
    python bert/collect_embeddings.py \
        --num-jobs 1 \
        --input-manifest data/librispeech_manifest/librispeech_cuts_${subset}.jsonl.gz \
        --manifest-name embeddings-${subset} \
        --target-manifest-file ${manifest_dir}/${model_name}-${model_version}-layer-${embedding_layer}-${subset}.jsonl.gz \
        --embedding-dir $embedding_dir \
        --embedding-layer $embedding_layer \
        --max-duration 200 \
        --bert-version $model_version
done