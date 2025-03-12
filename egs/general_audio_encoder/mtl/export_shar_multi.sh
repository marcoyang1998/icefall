#!/usr/bin/env bash

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

root_shar_dir=data-shar/data-shar-whisper-cb16-firered-cb16
mkdir -p $root_shar_dir
fbank_dir=data_s3/vq_whisper_turbo_cb16_firered_zh_en_cb16
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
                -c codebook_indexes:numpy \
                -c firered_codebook_indexes:numpy \
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
    for subset in full; do
        mkdir -p $shar_dir/$subset
        manifest=data/vq_dasheng_large_layer_-1_cb16/audioset_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting audioset ${subset}"
            lhotse shar export -j 8 \
                --audio wav \
                -c beats_embedding:numpy \
                -c audio_codebook_indexes:numpy \
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
    for subset in medium large; do
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

if [ $stage -le 50 ] && [ $stop_stage -ge 50 ]; then
    log "Processing WenetSpeech DEV"
    shar_dir=${root_shar_dir}/wenetspeech
    mkdir -p $shar_dir
    for subset in DEV; do
        manifest=$fbank_dir/wenetspeech_cuts_${subset}.jsonl.gz
        if [ ! -f $shar_dir/.shar.$subset.complete ]; then
            log "Start exporting wenetspeech ${subset}"
            lhotse shar export -j 8 \
                -c codebook_indexes:numpy \
                -c firered_codebook_indexes:numpy \
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
    for n in $(seq 0 1 9); do 
        manifest=$fbank_dir/wenetspeech_L_split/wenetspeech_cuts_L.0${n}.jsonl.gz
        if [ ! -f $shar_dir/.shar.${subset}.${n}.complete ]; then
            log "Start exporting wenetspeech ${subset} split ${n}"
            lhotse shar export -j 8 \
                -c codebook_indexes:numpy \
                -c firered_codebook_indexes:numpy \
                $manifest \
                $shar_dir/$subset/split_${n}
            touch $shar_dir/.shar.${subset}.${n}.complete
        fi
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Processing various chinese datasets"
    
    # for dataset in datatang1505 dialog dialog3k magicdata MagicData_dialog ximalaya aidatatang_200zh aishell3 aishell2 cs_wav acq zhvoice; do
    # for dataset in sensetime primewords_md_2018_set1 common_voice_20200622 accent baidu_en_cn cantonese; do
    # for dataset in digital_library_202003 ST-CMDS-20170001_1-OS en_us_english en8848 ljspeech tatoeba ted vctk voase voaSplider; do
    # for dataset in speech_annotations_2021; do
    # for dataset in speech_wav peoplespeech; do
    for dataset in phone 20220309; do
        shar_dir=${root_shar_dir}/$dataset
        mkdir -p $shar_dir
        manifest=$fbank_dir/${dataset}_cuts.jsonl.gz
        
        if [ ! -f $root_shar_dir/.shar.${dataset}.complete ]; then
            log "Start exporting ${dataset}"
            lhotse shar export -j 8 \
                -c codebook_indexes:numpy \
                $manifest \
                $shar_dir
            touch $root_shar_dir/.shar.${dataset}.complete
        fi
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Processing MLS"
    dataset=MLS
    shar_dir=${root_shar_dir}/${dataset}
    mkdir -p $shar_dir
    
    for n in $(seq 0 1 7); do 
        manifest=$fbank_dir/${dataset}_split/${dataset}_cuts.${n}.processed.jsonl.gz
        if [ ! -f $shar_dir/.shar.${dataset}.${n}.complete ]; then
            log "Start exporting $dataset split ${n}"
            lhotse shar export -j 8 \
                -c codebook_indexes:numpy \
                $manifest \
                $shar_dir/split_${n}
            touch $shar_dir/.shar.${dataset}.${n}.complete
        fi
    done
fi