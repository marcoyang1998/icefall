#!/usr/bin/env bash

export PYTHONPATH=./../../..:$PYTHONPATH

model_name=wavlm

manifest_dir=data/manifests/${model_name}
embedding_dir=data/embeddings/${model_name}
mkdir -p $manifest_dir
mkdir -p $embedding_dir

wavlm_version=large
embedding_layer=21

# for subset in dev-clean dev-other sampled; do
#     python wavlm/collect_embeddings.py \
#         --num-jobs 1 \
#         --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
#         --manifest-name embeddings-${subset} \
#         --target-manifest-file ${manifest_dir}/${model_name}-${model_version}-layer-${embedding_layer}-${subset}.jsonl.gz \
#         --embedding-dir $embedding_dir \
#         --embedding-layer $embedding_layer \
#         --max-duration 200 \
#         --wavlm-version $wavlm_version
# done

subset=vox1-10-speakers
python wavlm/collect_embeddings.py \
    --num-jobs 1 \
    --input-manifest vox1_test_10_speakers.jsonl.gz \
    --manifest-name embeddings-$subset \
    --target-manifest-file ${manifest_dir}/${model_name}-${model_version}-layer-${embedding_layer}-${subset}.jsonl.gz \
    --embedding-dir $embedding_dir \
    --embedding-layer $embedding_layer \
    --max-duration 200 \
    --wavlm-version $wavlm_version

# for subset in balanced; do
#     python zipformer_audio_encoder/collect_zipformer_embeddings.py \
#         --num-jobs 2 \
#         --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
#         --manifest-name embeddings-${subset} \
#         --target-manifest-file ${manifest_dir}/${model_name}-${model_version}-layer-${embedding_layer}-audioset-${subset}.jsonl.gz \
#         --model-ckpt $model_ckpt \
#         --num-encoder-layers 2,2,4,5,4,2 \
#         --feedforward-dim 512,1024,2048,3072,2048,1024 \
#         --encoder-dim 192,384,768,1024,768,384 \
#         --encoder-unmasked-dim 192,256,320,512,320,256 \
#         --embedding-dir $embedding_dir \
#         --embedding-layer $embedding_layer \
#         --max-duration 400 \
#         --zipformer-version $model_version
# done

