#!/usr/bin/env bash

stage=-1
stop_stage=-1

set -eou pipefail
. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

root_shar_dir=data-shar-no-feat
fbank_dir=data/vq_whisper_turbo_zh_en_16_v2
log "Fbank dir: $fbank_dir"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    shar_dir=${root_shar_dir}/librispeech
    mkdir -p $shar_dir
    for subset in dev-clean dev-other train-all-shuf; do
        mkdir -p $shar_dir/$subset
        manifest=$fbank_dir/librispeech_cuts_${subset}.jsonl.gz

        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting ${subset}"
            lhotse shar export -j 8 \
                --audio flac \
                -c codebook_indexes:numpy \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Processing audioset"
    shar_dir=${root_shar_dir}/audioset
    mkdir -p $shar_dir
    for subset in eval; do
        mkdir -p $shar_dir/$subset
        manifest=data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting audioset ${subset}"
            lhotse shar export -j 2 \
                --audio wav \
                -c beats_embedding:numpy \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Processing GigaSpeech"
    shar_dir=${root_shar_dir}/gigaspeech
    mkdir -p $shar_dir
    for subset in xl; do
        manifest=$fbank_dir/gigaspeech_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting gigaspeech ${subset}"
            lhotse shar export -j 8 \
                --audio wav \
                -c codebook_indexes:numpy \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Processing Libriheavy"
    shar_dir=${root_shar_dir}/libriheavy
    mkdir -p $shar_dir
    for subset in medium; do
        manifest=$fbank_dir/libriheavy_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting libriheavy ${subset}"
            lhotse shar export -j 8 \
                --audio flac \
                -c codebook_indexes:numpy \
                $manifest \
                $shar_dir/$subset
            touch $shar_dir/.shar.$subset.complete
        fi
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Processing WenetSpeech"
    shar_dir=${root_shar_dir}/wenetspeech
    mkdir -p $shar_dir
    
    subset=L
    for n in 1; do 
        # manifest=$fbank_dir/wenetspeech_cuts_${subset}.jsonl.gz
        manifest=$fbank_dir/wenetspeech_L_split/wenetspeech_cuts_L.${n}.processed.jsonl.gz
        if [ ! -f $shar_dir/.shar.${subset}.${n}.complete ]; then
            log "Start exporting wenetspeech ${subset} split ${n}"
            lhotse shar export -j 4 \
                --audio original \
                -c codebook_indexes:numpy \
                $manifest \
                $shar_dir/$subset/split_${n}
            touch $shar_dir/.shar.${subset}.${n}.complete
        fi
    done
fi