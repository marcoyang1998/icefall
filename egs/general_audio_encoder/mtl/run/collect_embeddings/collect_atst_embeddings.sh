#!/usr/bin/env bash

export PYTHONPATH=./../../..:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/dasheng_dev:$PYTHONPATH
export PYTHONPATH=/cpfs04/user/housiyuan/xiaoyu/workspace/audiossl:$PYTHONPATH

model_name=atst_frame

manifest_dir=data/manifests/${model_name}
embedding_dir=data/embeddings/${model_name}
mkdir -p $manifest_dir
mkdir -p $embedding_dir

embedding_layer=-1

# export CUDA_VISIBLE_DEVICES="0,1"
for subset in balanced eval; do
    python atst_frame/collect_embeddings.py \
        --num-jobs 2 \
        --input-manifest data/fbank_audioset/audioset_cuts_${subset}.jsonl.gz \
        --manifest-name audioset-${subset} \
        --target-manifest-file ${manifest_dir}/${model_name}-layer-${embedding_layer}-audioset-${subset}.jsonl.gz \
        --embedding-dir $embedding_dir \
        --embedding-layer $embedding_layer \
        --max-duration 200
done

# for subset in test-clean; do
#     python dasheng/collect_embeddings.py \
#         --num-jobs 1 \
#         --model-version $model_version \
#         --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
#         --manifest-name ls-${subset} \
#         --target-manifest-file ${manifest_dir}/${model_name}-layer-${embedding_layer}-ls-${subset}.jsonl.gz \
#         --embedding-dir $embedding_dir \
#         --embedding-layer $embedding_layer \
#         --max-duration 200
# done


# python dasheng/collect_embeddings.py \
#     --num-jobs 1 \
#     --model-version $model_version \
#     --input-manifest data/fbank_voxceleb/vox1_cuts_test.jsonl.gz \
#     --manifest-name vox1-test \
#     --target-manifest-file ${manifest_dir}/${model_name}-layer-${embedding_layer}-vox1-test.jsonl.gz \
#     --embedding-dir $embedding_dir \
#     --embedding-layer $embedding_layer \
#     --max-duration 200
