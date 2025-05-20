#!/usr/bin/env bash
export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/dasheng_dev:$PYTHONPATH

set -eou pipefail


stage=-1
stop_stage=-1

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

model_version=large
num_codebooks=8
embedding_layer=-1
normalize=0

# vq_dir=data/vq_dasheng_large_libri_as_cb_${num_codebooks}
vq_dir=data/vq_dasheng_large_layer_${embedding_layer}_normalize_${normalize}_cb_${num_codebooks}
mkdir -p $vq_dir
log "Manifest will be stored under: $vq_dir"

# quantizer_path=data/quantizer/dasheng-large-libri-as-cb-${num_codebooks}.pt
quantizer_path=data/quantizer/dasheng-large-layer-${embedding_layer}-normalize-${normalize}-cb-${num_codebooks}.pt
log "Using ${quantizer_path}"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python dasheng/train_mvq.py \
        --embedding-dim 1536 \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/dasheng/dasheng-layer--1-audioset-balanced.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/dasheng/dasheng-layer--1-audioset-eval.jsonl.gz
            
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
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on audioset sets"
    for subset in full; do
        python dasheng/extract_mvq.py \
            --num-jobs 6 \
            --model-version $model_version \
            --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer ${embedding_layer} \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Collect MVQ tokens on LibriSpeech training sets"
    for subset in dev-clean dev-other; do
        python dasheng/extract_mvq.py \
            --num-jobs 1 \
            --model-version $model_version \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer ${embedding_layer} \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
    
    
    for subset in train-all-shuf; do
        python dasheng/extract_mvq.py \
            --num-jobs 4 \
            --model-version $model_version \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer ${embedding_layer} \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done

fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Collect MVQ tokens on ESC"
    python dasheng/extract_mvq.py \
        --num-jobs 1 \
        --model-version $model_version \
        --input-manifest data/esc/esc_cuts.jsonl.gz \
        --target-manifest-file $vq_dir/esc_cuts.jsonl.gz \
        --embedding-dim 1536 \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-esc \
        --embedding-dir $vq_dir \
        --embedding-layer ${embedding_layer} \
        --quantizer-path $quantizer_path \
        --max-duration 200
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Collect MVQ tokens on Vox1-test"
    python dasheng/extract_mvq.py \
        --num-jobs 1 \
        --model-version $model_version \
        --input-manifest data/fbank_voxceleb/vox1_cuts_test.jsonl.gz \
        --target-manifest-file $vq_dir/vox1_test_cuts.jsonl.gz \
        --embedding-dim 1536 \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-vox1-test \
        --embedding-dir $vq_dir \
        --embedding-layer ${embedding_layer} \
        --quantizer-path $quantizer_path \
        --max-duration 200
fi