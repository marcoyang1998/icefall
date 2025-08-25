#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

stage=-1
stop_stage=-1

set -eou pipefail
. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

vq_name=vq_hubert_large_layer_21_normalize_1_cb_16
log "vq folder: ${vq_name}"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for subset in dev-clean dev-other train-all-shuf; do
        python local/npy2hdf5.py \
            --num-jobs 8 \
            --input-manifest data/${vq_name}/librispeech_cuts_${subset}.jsonl.gz \
            --manifest-name librispeech_cuts_$subset \
            --manifest-dir data_hdf5/${vq_name} \
            --target-manifest-file data_hdf5/${vq_name}/librispeech_cuts_${subset}.jsonl.gz
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for subset in dev xs s m l xl; do
        python local/npy2hdf5.py \
            --num-jobs 12 \
            --input-manifest data/${vq_name}/gigaspeech_cuts_${subset}.jsonl.gz \
            --manifest-name gigaspeech_cuts_$subset \
            --manifest-dir data_hdf5/${vq_name} \
            --target-manifest-file data_hdf5/${vq_name}/gigaspeech_cuts_${subset}.jsonl.gz
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    for subset in small medium large; do
        python local/npy2hdf5.py \
            --num-jobs 12 \
            --input-manifest data/${vq_name}/libriheavy_cuts_${subset}.jsonl.gz \
            --manifest-name libriheavy_cuts_$subset \
            --manifest-dir data_hdf5/${vq_name} \
            --target-manifest-file data_hdf5/${vq_name}/libriheavy_cuts_${subset}.jsonl.gz
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Processing msp-podcast"
    for subset in Train Development Test1 Test2; do
        log "Exporting msp-podcast ${subset}"
        python local/npy2hdf5.py \
            --num-jobs 1 \
            --input-manifest data/${vq_name}/msp_podcast_cuts_${subset}.jsonl.gz \
            --manifest-name podcast_cuts_$subset \
            --manifest-dir data_hdf5/${vq_name} \
            --target-manifest-file data_hdf5/${vq_name}/msp_podcast_cuts_${subset}.jsonl.gz
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Processing MELD"
    for subset in dev test train; do
        log "Exporting MELD ${subset}"
        python local/npy2hdf5.py \
            --num-jobs 1 \
            --input-manifest data/${vq_name}/meld_cuts_${subset}.jsonl.gz \
            --manifest-name podcast_cuts_$subset \
            --manifest-dir data_hdf5/${vq_name} \
            --target-manifest-file data_hdf5/${vq_name}/meld_cuts_${subset}.jsonl.gz
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    for dataset in iemocap mead; do
        log "Exporting ${dataset}"
        python local/npy2hdf5.py \
            --num-jobs 1 \
            --input-manifest data/${vq_name}/${dataset}_cuts_all.jsonl.gz \
            --manifest-name podcast_cuts_$subset \
            --manifest-dir data_hdf5/${vq_name} \
            --target-manifest-file data_hdf5/${vq_name}/${dataset}_cuts_all.jsonl.gz
    done
fi