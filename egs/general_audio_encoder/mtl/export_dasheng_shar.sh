#!/usr/bin/env bash

export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH

stage=-1
stop_stage=-1
num_codebooks=16

set -eou pipefail
. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

root_shar_dir=data-shar/data-shar-whisper-zh-en-cb16-v2
mkdir -p $root_shar_dir
fbank_dir=data/vq_dasheng_large_cb_16

log "Shar dir: $root_shar_dir"
log "Fbank dir: $fbank_dir"


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Processing audioset"
    shar_dir=${root_shar_dir}/audioset
    mkdir -p $shar_dir
    for subset in balanced eval full; do
        mkdir -p $shar_dir/$subset
        manifest=data/vq_dasheng_large_cb_16/audioset_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting audioset ${subset}"
            lhotse shar export -j 8 \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi
