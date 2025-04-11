#!/usr/bin/env bash
export PYTHONPATH=./../../../:$PYTHONPATH

set -eou pipefail

num_codebooks=16

stage=-1
stop_stage=-1

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

model_version=large
vq_dir=data/vq_dasheng_large_cb_${num_codebooks}
mkdir -p $vq_dir

quantizer_path=data/quantizer/dasheng-large-cb-${num_codebooks}.pt

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python dasheng/train_mvq.py \
        --embedding-dim 1536 \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        data/manifests/dasheng/dasheng-layer--1-audioset-balanced.jsonl.gz      
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on audioset sets"
    for subset in balanced eval; do
        python dasheng/extract_mvq.py \
            --num-jobs 2 \
            --model-version $model_version \
            --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on audioset sets"
    for subset in full; do
        python dasheng/extract_mvq.py \
            --num-jobs 8 \
            --model-version $model_version \
            --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi