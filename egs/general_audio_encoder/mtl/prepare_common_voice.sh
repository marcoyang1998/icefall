#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

# set -euxo pipefail

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

log "Running prepare_common_voice.sh"

log "dl_dir: $dl_dir"

# First, download the common voice dataset

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Download dataset and unzip"
  
  subset=train
  for lang in "$dl_dir"/common_voice_17_0/audio/*; do
    lang=$(basename "$lang")
    if [ ! -e "$dl_dir"/common_voice_17_0/audio/$lang/.untar.done ]; then
      log "Untarring $lang"
      if [ -d "$dl_dir"/common_voice_17_0/audio/$lang/$subset ]; then
        cd $dl_dir/common_voice_17_0/audio/$lang/$subset/
        files=$(find . -name "*.tar")
        for file in $files; do
          echo "Untarring ${file}"
          tar -xf $file
        done
        cd -
        touch "$dl_dir"/common_voice_17_0/audio/$lang/.untar.done
      fi
    fi
  done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Convert the mp3 files to wav files for CV"
    for lang in "$dl_dir"/common_voice_17_0/audio/*; do
      lang=$(basename "$lang")
      log "Converting $lang to wav"
      for subset in train; do
        python local/convert_mp3_to_wav.py $dl_dir/common_voice_17_0/audio/$lang/$subset 
      done
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Create common voice manifest"
  # We assume that you have downloaded the MLS corpus
  # to $dl_dir/MLS
  for lang in en; do
    if [ ! -e data/cv17_manifest/.manifest.done ]; then
      for subset in train; do
        python local/generate_common_voice_manifest.py \
          --dataset-dir download/common_voice_17_0 \
          --manifest-dir data/cv17_manifest \
          --subset $subset \
          --language $lang
      done
    fi
  done
fi