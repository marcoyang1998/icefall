#!/usr/bin/env bash

firered_root=/fs-computility/INTERN6/shared/yangxiaoyu/workspace/FireRedASR
export PATH=$firered_root/fireredasr/:$firered_root/fireredasr/utils/:$PATH
export PYTHONPATH=$firered_root/:$PYTHONPATH
export PYTHONPATH=./../../..:$PYTHONPATH


model_name=firered

manifest_dir=data/manifests/${model_name}
embedding_dir=data/embeddings/${model_name}
mkdir -p $manifest_dir
mkdir -p $embedding_dir

embedding_layer=-1
model_dir=$firered_root/pretrained_models/FireRedASR-AED-L

# for subset in wenetspeech_subset giga_subset libri_mix aishell_subset; do
# for subset in wenetspeech_subset_trimmed; do
#     python firered/collect_embeddings.py \
#         --num-jobs 1 \
#         --model-dir $model_dir \
#         --input-manifest data/manifests/${subset}_cuts.jsonl.gz \
#         --manifest-name embeddings-${subset} \
#         --target-manifest-file ${manifest_dir}/${model_name}-layer-${embedding_layer}-${subset}.jsonl.gz \
#         --embedding-dir $embedding_dir \
#         --embedding-layer $embedding_layer \
#         --max-duration 200
# done

export CUDA_VISIBLE_DEVICES="2"
for subset in s; do
    python firered/collect_embeddings.py \
        --num-jobs 1 \
        --model-dir $model_dir \
        --input-manifest data/manifests/libri_mix_20k_cuts.jsonl.gz \
        --manifest-name embeddings-libri-20k \
        --target-manifest-file ${manifest_dir}/${model_name}-layer-${embedding_layer}-libri-mix-20k.jsonl.gz \
        --embedding-dir $embedding_dir \
        --embedding-layer $embedding_layer \
        --max-duration 200
done
