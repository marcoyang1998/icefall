#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/dasheng_dev:$PYTHONPATH
export PYTHONPATH=/cpfs04/user/housiyuan/xiaoyu/workspace/audiossl:$PYTHONPATH

set -eou pipefail


stage=-1
stop_stage=-1

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

model_version=base
num_codebooks=8
embedding_layer=-1
normalize=0

vq_dir=data/vq_atst_frame_layer_${embedding_layer}_normalize_${normalize}_cb_${num_codebooks}
mkdir -p $vq_dir
log "Manifest will be stored under: $vq_dir"

# quantizer_path=data/quantizer/dasheng-large-libri-as-cb-${num_codebooks}.pt
quantizer_path=data/quantizer/atst-frame-layer-${embedding_layer}-normalize-${normalize}-cb-${num_codebooks}.pt
log "Using ${quantizer_path}"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python atst_frame/train_mvq.py \
        --embedding-dim 768 \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/atst_frame/atst_frame-layer--1-audioset-balanced.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/atst_frame/atst_frame-layer--1-audioset-eval.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on audioset sets"
    for subset in balanced eval; do
        python atst_frame/extract_mvq_hdf5.py \
            --num-jobs 2 \
            --model-version $model_version \
            --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --embedding-dim 768 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on audioset sets"
    for subset in full; do
        python atst_frame/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --model-version $model_version \
            --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --embedding-dim 768 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer ${embedding_layer} \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi
