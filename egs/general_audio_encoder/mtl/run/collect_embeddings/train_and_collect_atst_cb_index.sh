#!/usr/bin/env bash

cd /mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl
echo "Current dir: $PWD"

####### Mount the necessary disks #######
bash mount_brainllm_h.sh
ls -lh download/LibriSpeech
#########################################

############## PYTHON env ###############
source /home/housiyuan/miniconda3/etc/profile.d/conda.sh && conda activate encoder

work_dir=/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl
cd $work_dir

echo "Current Directory: $PWD"

export PYTHONPATH=./../../..:$PYTHONPATH
export PYTHONPATH=/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH
export PYTHONPATH=/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/audiossl:$PYTHONPATH

#########################################

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
num_codebooks=16
embedding_layer=-1
concat_all_layers=1
normalize=0

if [ "$concat_all_layers" -eq 1 ]; then
    embedding_dim=$(( 768 * 12 ))
else
    embedding_dim=768
fi

vq_dir=data/vq_atst_frame_layer_${embedding_layer}_concat_${concat_all_layers}_normalize_${normalize}_cb_${num_codebooks}
mkdir -p $vq_dir
log "Manifest will be stored under: $vq_dir"

# quantizer_path=data/quantizer/dasheng-large-libri-as-cb-${num_codebooks}.pt
quantizer_path=data/quantizer/atst-frame-layer-${embedding_layer}-concat-${concat_all_layers}-normalize-${normalize}-cb-${num_codebooks}.pt
log "Using ${quantizer_path}"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python atst_frame/train_mvq.py \
        --embedding-dim $embedding_dim \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/atst_frame/atst_frame-layer--1-concat-all-1-audioset-balanced.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/atst_frame/atst_frame-layer--1-concat-all-1-audioset-eval.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on audioset sets"
    for subset in balanced eval; do
        python atst_frame/extract_mvq_hdf5.py \
            --num-jobs 2 \
            --model-version $model_version \
            --input-manifest data/audioset_manifest/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --concat-all-layers $concat_all_layers \
            --embedding-dim $embedding_dim \
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
            --input-manifest data/audioset_manifest/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --concat-all-layers $concat_all_layers \
            --embedding-dim $embedding_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer ${embedding_layer} \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi
