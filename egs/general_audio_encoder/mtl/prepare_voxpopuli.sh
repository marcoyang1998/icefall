#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

set -euxo pipefail

nj=20
stage=-1
stop_stage=100

. shared/parse_options.sh || exit 1

task=asr
dl_dir=download

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Download data"
    # for lang in "en" "de" "fr" "es" "pl" "it" "ro" "hu" "cs" "nl" "fi" "hr" "sk" "sl" "et" "lt"; do
    # for lang in "cs" "nl" "fi" "hr" "sk" "sl" "et" "lt"; do
    # for lang in "pt" "bg" "el" "lv" "mt" "sv" "da"; do
    for lang in en_v2; do
        if [ ! -d $dl_dir/voxpopuli/raw_audios/${lang} ]; then
            log "Downloading ${lang}"
            lhotse download voxpopuli --subset $lang $dl_dir/voxpopuli_en
        fi
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare VoxPopuli manifest"
  # We assume that you have downloaded the VoxPopuli corpus
  # to $dl_dir/voxpopuli
  # for lang in en de fr es pl it ro hu; do
  # for lang in "cs" "nl" "fi" "hr" "sk" "sl" "et" "lt"; do
  for lang in "pt" "bg" "el" "lv" "mt" "sv" "da"; do
    if [ ! -e data/manifests/.voxpopuli-${task}-${lang}.done ]; then
        # Warning : it requires Internet connection (it downloads transcripts to ${tmpdir})
        lhotse prepare voxpopuli --task asr --lang $lang -j $nj $dl_dir/voxpopuli data/voxpopuli_manifest
        touch data/manifests/.voxpopuli-${task}-${lang}.done
    fi
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Preprocess VoxPopuli manifest"
  mkdir -p data/voxpopuli_manifest
  
  # recordings + supervisions -> cutset
  # for lang in en de fr es pl it ro hu; do
  # for lang in "cs" "nl" "fi" "hr" "sk" "sl" "et" "lt"; do
  for lang in "pt" "bg" "el" "lv" "mt" "sv" "da"; do
    if [ ! -e data/voxpopuli_manifest/.voxpopuli-${task}-${lang}-preprocess_complete ]; then
      python ./local/preprocess_voxpopuli.py --task $task --lang $lang \
          --use-original-text True
      touch data/fbank/.voxpopuli-${task}-${lang}-preprocess_complete
    fi
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Trim the Voxpupuli cuts to supervisions"
  output_manifest=data_hdf5/voxpopuli_manifest_trimmed
  mkdir -p $output_manifest
  
  # for lang in en de fr es pl it ro hu; do
  for lang in "cs" "nl" "fi" "hr" "sk" "sl" "et" "lt"; do
    for subset in train dev test; do
      if [ ! -e $output_manifest/voxpopuli-asr-${lang}_cuts_${subset}.jsonl.gz ]; then
        python ./local/trim2supervision.py \
          --input-manifest data/voxpopuli_manifest/voxpopuli-asr-${lang}_cuts_${subset}_raw.jsonl.gz \
          --output-manifest ${output_manifest}/voxpopuli-asr-${lang}_cuts_${subset}.jsonl.gz
      fi
    done
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 0: Download unlabelled data"
    for lang in en_v2; do
        if [ ! -d $dl_dir/voxpopuli/raw_audios/${lang} ]; then
            log "Downloading ${lang}"
            lhotse download voxpopuli --subset $lang $dl_dir/voxpopuli_en
        fi
    done
fi
