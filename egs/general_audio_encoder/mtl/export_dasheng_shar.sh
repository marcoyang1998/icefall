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

root_shar_dir=data-shar/data-shar-dasheng-as-cb8
mkdir -p $root_shar_dir
vq_dir=data/vq_dasheng_large_layer_-1_normalize_0_cb_8

log "Shar dir: $root_shar_dir"
log "vq dir: $vq_dir"


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Processing audioset"
    shar_dir=${root_shar_dir}/audioset
    mkdir -p $shar_dir
    for subset in balanced eval full; do
        mkdir -p $shar_dir/$subset
        manifest=$vq_dir/audioset_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting audioset ${subset}"
            lhotse shar export -j 8 \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Processing music4all"
    shar_dir=${root_shar_dir}/music4all
    mkdir -p $shar_dir
    for subset in all; do
        mkdir -p $shar_dir/$subset
        manifest=$vq_dir/music4all_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting music4all ${subset}"
            lhotse shar export -j 8 \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Processing BBC soundeffect"
    shar_dir=${root_shar_dir}/bbc_soundeffect
    mkdir -p $shar_dir
    for subset in train_10s test_10s; do
        mkdir -p $shar_dir/$subset
        manifest=$vq_dir/bbc_soundeffect_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting bbc_soundeffect ${subset}"
            lhotse shar export -j 8 \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Processing VGG sound"
    shar_dir=${root_shar_dir}/vggsound
    mkdir -p $shar_dir
    for subset in train test; do
        mkdir -p $shar_dir/$subset
        manifest=$vq_dir/vggsound_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting vgg sound ${subset}"
            lhotse shar export -j 8 \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Processing Freesound"
    shar_dir=${root_shar_dir}/freesound
    mkdir -p $shar_dir
    for subset in train_10s test_10s; do
        mkdir -p $shar_dir/$subset
        manifest=$vq_dir/freesound_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting vgg sound ${subset}"
            lhotse shar export -j 8 \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Processing MTG"
    shar_dir=${root_shar_dir}/mtg_wav
    mkdir -p $shar_dir
    
    manifest=$vq_dir/mtg_wav_cuts_10s.jsonl.gz
    if [ ! -f $root_shar_dir/.mtg_wav.complete ]; then
        log "Start exporting MTG cuts"
        lhotse shar export -j 8 \
            $manifest \
            $shar_dir/
        touch $shar_dir/.mtg_wav.complete
    fi
fi