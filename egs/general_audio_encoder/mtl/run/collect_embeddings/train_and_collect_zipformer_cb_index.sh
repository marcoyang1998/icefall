#!/usr/bin/env bash
export PYTHONPATH=./../../../:$PYTHONPATH

set -eou pipefail

stage=-1
stop_stage=-1

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

model_name=zipformer
model_version=300m-lh-large-pretrained
model_dim=1024
embedding_layer=5
num_codebooks=16
normalize=1

vq_dir=data/vq_zipformer_${model_version}_layer_${embedding_layer}_normalize_${normalize}_cb_${num_codebooks}
mkdir -p $vq_dir

model_ckpt=/cpfs02/user/housiyuan/xiaoyu/workspace/icefall_general_audio_encoder/egs/general_audio_encoder/mtl/zipformer_audio_encoder/exp-300M-zipformer-non-streaming-lh-large-out-ds-2-mask-ratio-1.0-musan-1-rir-0-hubert-large-layer-21-normalized-mvq-cb16-shar/iter-400000-avg-4.pt
quantizer_path=data/quantizer/zipformer-${model_version}-layer-${embedding_layer}-normalize-${normalize}-cb-${num_codebooks}.pt
prefix_folder=/cpfs02/user/housiyuan/xiaoyu/codebook_indexes/zipformer_${model_version}_layer_${embedding_layer}_normalized_cb_${num_codebooks}

log "VQ dir: $vq_dir"
log "Quantizer: $quantizer_path"
log "Prefix: $prefix_folder"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python local/train_mvq.py \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/zipformer/zipformer-300m-lh-large-pretrained-layer-5-sampled.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/zipformer/zipformer-300m-lh-large-pretrained-layer-5-dev-clean.jsonl.gz \
            data/manifests/zipformer/zipformer-300m-lh-large-pretrained-layer-5-dev-other.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech training sets"
    for subset in dev-clean dev-other; do
        python zipformer_audio_encoder/extract_mvq.py \
            --num-jobs 1 \
            --input-manifest  data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --zipformer-version $model_version \
            --model-ckpt $model_ckpt \
            --embedding-dim $model_dim \
            --num-cb $num_codebooks \
            --num-encoder-layers 2,2,4,5,4,2 \
            --feedforward-dim 512,1024,2048,3072,2048,1024 \
            --encoder-dim 192,384,768,1024,768,384 \
            --encoder-unmasked-dim 192,256,320,512,320,256 \
            --manifest-name codebook-indexes-libri-${subset} \
            --s3-prefix $prefix_folder/LibriSpeech/${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 400
    done

    for subset in train-all-shuf; do
        python zipformer_audio_encoder/extract_mvq.py \
            --num-jobs 4 \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --zipformer-version $model_version \
            --model-ckpt $model_ckpt \
            --embedding-dim $model_dim \
            --num-cb $num_codebooks \
            --num-encoder-layers 2,2,4,5,4,2 \
            --feedforward-dim 512,1024,2048,3072,2048,1024 \
            --encoder-dim 192,384,768,1024,768,384 \
            --encoder-unmasked-dim 192,256,320,512,320,256 \
            --manifest-name codebook-indexes-libri-${subset} \
            --s3-prefix $prefix_folder/LibriSpeech/${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 400
    done
fi