#!/usr/bin/env bash

firered_root=/mnt/cache/share_data/housiyuan/FireRedASR

export PATH=$firered_root/fireredasr/:$firered_root/fireredasr/utils/:$PATH
export PYTHONPATH=$firered_root/:$PYTHONPATH
export PYTHONPATH=./../../..:$PYTHONPATH

export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH

set -eou pipefail

num_codebooks=16

stage=-1
stop_stage=-1

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

model_dir=$firered_root/pretrained_models/FireRedASR-AED-L

vq_dir=data/vq_firered_zh_en_${num_codebooks}_v2
mkdir -p $vq_dir

quantizer_path=data/quantizer/firered-zh-en-cb-${num_codebooks}-v2.pt

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python firered/train_mvq.py \
        --embedding-dim 1280 \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        data/manifests/firered/firered-layer--1-wenet-trimmed-S.jsonl.gz \
        data/manifests/firered/firered-layer--1-aishell-train-70k.jsonl.gz \
        data/manifests/firered/firered-layer--1-giga-s-70k.jsonl.gz \
        data/manifests/firered/firered-layer--1-libri-mix-20k.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech dev/test sets"
    for subset in dev-clean dev-other; do
        python firered/extract_mvq.py \
            --num-jobs 1 \
            --model-dir $model_dir \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on LibriSpeech training sets"
    for subset in train-all-shuf; do
        python firered/extract_mvq.py \
            --num-jobs 8 \
            --model-dir $model_dir \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Stage 10: Collect MVQ tokens on wenetspeech L"
    
    subset=L
    num_splits=10
    split_dir=$vq_dir/wenetspeech_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split --no-pad $num_splits data/fbank_wenetspeech_wav_trimmed/wenetspeech_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 9); do
        log "Start encoding wenetspeech L split ${i}"
        if [ ! -f  $split_dir/wenetspeech_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python firered/extract_mvq.py \
                --num-jobs 8 \
                --model-dir $model_dir \
                --input-manifest $split_dir/wenetspeech_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/wenetspeech_cuts_${subset}.${i}.processed.jsonl.gz \
                --embedding-dim 1280 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-wenetspeech-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer -1 \
                --quantizer-path $quantizer_path \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/wenetspeech_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of wenetspeech L"
        pieces=$(find $split_dir -name "wenetspeech_cuts_L.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/wenetspeech_cuts_L.jsonl.gz
    fi
fi

# export CUDA_VISIBLE_DEVICES="2,3"
if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Stage 11: Collect MVQ tokens on WeneSpeech test/dev"
    for subset in DEV; do
        log "Processing $subset"
        python firered/extract_mvq.py \
            --num-jobs 2 \
            --model-dir $model_dir \
            --input-manifest data/fbank_wenetspeech_wav_trimmed/wenetspeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/wenetspeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-wenetspeech-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    log "Stage 12: Collect MVQ tokens on various Chinese datasets"
    
    # for dataset in aidatatang_200zh aishell3 cs_wav acq aishell2 baidu_en_cn cantonese accent \
    #     aishell2 baidu_en_cn cantonese accent datatang1505 dialog dialog3k magicdata \
    #     MagicData_dialog ximalaya zhvoice primewords_md_2018_set1 phone speech_wav; do
    for dataset in speech_wav peoplespeech digital_library_202003 ST-CMDS-20170001_1-OS en_us_english en8848 \
        ljspeech tatoeba ted vctk voase voaSplider 20220309; do

        log "Processing $dataset"
        mkdir -p $vq_dir/${dataset}
        python firered/extract_mvq.py \
            --num-jobs 8 \
            --input-manifest ASR_data/preprocessed_manifest/${dataset}_cuts.jsonl.gz \
            --target-manifest-file $vq_dir/${dataset}_cuts.jsonl.gz \
            --model-dir $model_dir \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-${dataset} \
            --embedding-dir $vq_dir/${dataset} \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi


if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
    log "Stage 13: Collect MVQ tokens on large chinese data"
    
    for subset in speech_annotations_2021; do
        num_splits=8
        split_dir=$vq_dir/${subset}_split
        mkdir -p $split_dir

        log "Splitting $subset"
        if [ ! -f $split_dir/.split_completed ]; then
            lhotse split $num_splits ASR_data/preprocessed_manifest/${subset}_cuts.jsonl.gz $split_dir
            touch $split_dir/.split_completed
        fi
        
        for i in $(seq 0 1 $(($num_splits-1))); do
            log "Start encoding ${subset} split ${i}"
            if [ ! -f  $split_dir/${subset}_cuts.${i}.processed.jsonl.gz ]; then
                python firered/extract_mvq.py \
                    --num-jobs 8 \
                    --input-manifest $split_dir/${subset}_cuts.${i}.jsonl.gz \
                    --target-manifest-file $split_dir/${subset}_cuts.${i}.processed.jsonl.gz \
                    --model-dir $model_dir \
                    --embedding-dim 1280 \
                    --num-codebooks $num_codebooks \
                    --manifest-name codebook-indexes-${subset}-split-${i} \
                    --embedding-dir $split_dir \
                    --embedding-layer -1 \
                    --quantizer-path $quantizer_path \
                    --max-duration 200
            fi
        done

        if [ ! -f $vq_dir/${subset}_cuts.jsonl.gz ]; then
            log "Combining the processed cuts of ${subset}"
            pieces=$(find $split_dir -name "${subset}_cuts.*.processed.jsonl.gz")
            lhotse combine $pieces $vq_dir/${subset}_cuts.jsonl.gz
        fi
    done
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
    log "Stage 14: Collect MVQ tokens on wechat reading datasets"
    
    for split in $(seq 5 1 9); do
        dataset=weread-16k-res-0${split}
        log "Processing $dataset"
        mkdir -p $vq_dir/${dataset}
        python firered/extract_mvq.py \
            --num-jobs 8 \
            --input-manifest ASR_data/preprocessed_manifest/${dataset}_cuts.jsonl.gz \
            --target-manifest-file $vq_dir/${dataset}_cuts.jsonl.gz \
            --model-dir $model_dir \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-${dataset} \
            --embedding-dir $vq_dir/${dataset} \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
    log "Stage 15: Collect MVQ tokens on Libriheavy small"
    
    subset=small
    num_splits=4
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits data/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy small split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python firered/extract_mvq.py \
                --num-jobs 4 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --model-dir $model_dir \
                --embedding-dim 1280 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer -1 \
                --quantizer-path $quantizer_path \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy small"
        pieces=$(find $split_dir -name "libriheavy_cuts_small.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_small.jsonl.gz
    fi
fi

if [ $stage -le 16 ] && [ $stop_stage -ge 16 ]; then
    log "Stage 16: Collect MVQ tokens on Libriheavy medium"
    
    subset=medium
    num_splits=5
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits data_s3/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy medium split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python firered/extract_mvq.py \
                --num-jobs 1 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --model-dir $model_dir \
                --embedding-dim 1280 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer -1 \
                --quantizer-path $quantizer_path \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy medium"
        pieces=$(find $split_dir -name "libriheavy_cuts_medium.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_medium.jsonl.gz
    fi
fi