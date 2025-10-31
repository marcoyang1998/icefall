#!/usr/bin/env bash

cd /mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl
echo "Current dir: $PWD"

####### Mount the necessary disks #######
bash mount_brainllm_h.sh
#########################################

source /home/housiyuan/miniconda3/etc/profile.d/conda.sh && conda activate encoder
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

model_version=large
model_dim=1024
embedding_layer=-1
num_codebooks=8
normalize=0

vq_dir=data_hdf5/vq_bert_${model_version}_layer_${embedding_layer}_normalize_${normalize}_libri_cb_${num_codebooks}
# vq_dir=data_hdf5/vq_bert_cb${num_codebooks}_dasheng_cb8_combined
mkdir -p $vq_dir


quantizer_path=data/quantizer/bert-${model_version}-layer-${embedding_layer}-normalize-${normalize}-libri-cb-${num_codebooks}.pt

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python bert/train_mvq.py \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/bert/bert-${model_version}-layer-${embedding_layer}-train-all-shuf.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/bert/bert-large-layer-${embedding_layer}-dev-clean.jsonl.gz \
            data/manifests/bert/bert-large-layer-${embedding_layer}-dev-other.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech training sets"
    for subset in dev-clean dev-other train-all-shuf; do
        embedding_dir=$vq_dir/librispeech_${subset}
        mkdir -p $embedding_dir
        if [ "$subset" == "train-all-shuf" ]; then
            num_gpus=8
        else
            num_gpus=1
        fi
        python bert/extract_mvq_hdf5.py \
            --num-jobs $num_gpus \
            --input-manifest data/librispeech_manifest/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --bert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi