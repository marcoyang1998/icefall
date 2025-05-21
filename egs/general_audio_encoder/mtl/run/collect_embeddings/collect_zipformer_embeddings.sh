#!/usr/bin/env bash

export PYTHONPATH=./../../..:$PYTHONPATH

model_name=zipformer

manifest_dir=data/manifests/${model_name}
embedding_dir=data/embeddings/${model_name}
mkdir -p $manifest_dir
mkdir -p $embedding_dir

model_ckpt=zipformer_audio_encoder/exp-300M-zipformer-non-streaming-lh-large-out-ds-2-mask-ratio-1.0-musan-1-rir-0-hubert-large-layer-21-normalized-mvq-cb16-shar/iter-400000-avg-4.pt
model_version=300m-lh-large-pretrained
embedding_layer=5

for subset in test-clean dev-clean dev-other sampled; do
    python zipformer_audio_encoder/collect_zipformer_embeddings.py \
        --num-jobs 1 \
        --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
        --manifest-name embeddings-${subset} \
        --target-manifest-file ${manifest_dir}/${model_name}-${model_version}-layer-${embedding_layer}-${subset}.jsonl.gz \
        --model-ckpt $model_ckpt \
        --num-encoder-layers 2,2,4,5,4,2 \
        --feedforward-dim 512,1024,2048,3072,2048,1024 \
        --encoder-dim 192,384,768,1024,768,384 \
        --encoder-unmasked-dim 192,256,320,512,320,256 \
        --embedding-dir $embedding_dir \
        --embedding-layer $embedding_layer \
        --max-duration 400 \
        --zipformer-version $model_version
done

