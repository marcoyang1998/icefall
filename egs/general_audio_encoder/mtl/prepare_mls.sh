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

log "Running prepare.sh"

log "dl_dir: $dl_dir"

languages=(
  english 
  german
  dutch
  spanish
  italian
  french
  polish
  portuguese
)

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare MLS manifest"
  # We assume that you have downloaded the MLS corpus
  # to $dl_dir/MLS
  mkdir -p data/mls_manifest
  if [ ! -e data/mls_manifest/.mls.done ]; then
    lhotse prepare mls -j $nj --flac $dl_dir/mls data/mls_manifest
    touch data/mls_manifest/.mls.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Create MLS manifest"
  # We assume that you have downloaded the MLS corpus
  # to $dl_dir/MLS
  
  if [ ! -e data/mls_manifest/.mls_manifest.done ]; then
    for lang in "dutch" "german" "spanish" "french" "italian" "polish" "portuguese"; do
      python local/combine_mls_manifest.py \
        --part $lang \
        --manifest-dir data/mls_manifest
    done
  fi
fi