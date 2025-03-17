#!/usr/bin/env bash
icefall_root=$(realpath ../../..)
export PYTHONPATH=${icefall_root}:$PYTHONPATH
export PYTHONPATH=/mnt/petrelfs/share_data/housiyuan/lhotse:$PYTHONPATH

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

vq_dir=data/vq_whisper_turbo_zh_en_${num_codebooks}_v2_numpy
mkdir -p $vq_dir

quantizer_path=data/quantizer/whisper-turbo-zh-en-cb-${num_codebooks}-v2.pt

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python whisper/train_mvq.py \
        --embedding-dim 1280 \
        --num-codebooks $num_codebooks \
        --feature-type h5 \
        --quantizer-path $quantizer_path \
        data/manifests/aishell_subset-whisper-turbo-layer--1.jsonl.gz \
        data/manifests/libri_giga_wenet_mix-whisper-turbo-layer--1.jsonl.gz
fi
        

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech dev/test sets"

    for subset in dev-clean dev-other; do
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 1 \
            --input-manifest data_s3/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --s3-prefix "brainllm:s3://yangxiaoyu/LibriSpeech/${subset}" \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 250
    done
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech dev/test sets"
    # --input-manifest data_s3/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
    for subset in small; do
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 1 \
            --input-manifest data_s3/fbank_libriheavy/libriheavy_cuts_small.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --s3-prefix "brainllm:s3://yangxiaoyu/librilight" \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 250
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on LibriSpeech training sets"
    for subset in train-all-shuf; do
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 8 \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Collect MVQ tokens on GigaSpeech"
    for subset in s; do
        log "Processing $subset"
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 2 \
            --input-manifest data/fbank_gigaspeech/gigaspeech_cuts_${subset}_raw.jsonl.gz \
            --target-manifest-file $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-giga-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Collect MVQ tokens on GigaSpeech"
    for subset in m l; do
        log "Processing $subset"
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 8 \
            --input-manifest data/fbank_gigaspeech/gigaspeech_cuts_${subset}_raw.jsonl.gz \
            --target-manifest-file $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-giga-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Collect MVQ tokens on GigaSpeech xl"
    
    subset=xl
    num_splits=6
    split_dir=$vq_dir/giga_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits data/fbank_gigaspeech/gigaspeech_cuts_${subset}_raw.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding giga xl split ${i}"
        if [ ! -f  $split_dir/gigaspeech_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python whisper/extract_whisper_mvq_client.py \
                --num-jobs 8 \
                --input-manifest $split_dir/gigaspeech_cuts_${subset}_raw.${i}.jsonl.gz \
                --target-manifest-file $split_dir/gigaspeech_cuts_${subset}.${i}.processed.jsonl.gz \
                --n-mels 128 \
                --embedding-dim 1280 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-giga-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer -1 \
                --quantizer-path $quantizer_path \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of gigaspeech xl"
        pieces=$(find $split_dir -name "gigaspeech_cuts_xl.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/gigaspeech_cuts_xl.jsonl.gz
    fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Collect MVQ tokens on Libriheavy small"
    
    subset=small
    num_splits=4
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data_s3/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy small split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python whisper/extract_whisper_mvq_client.py \
                --num-jobs 4 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --n-mels 128 \
                --embedding-dim 1280 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --s3-prefix "brainllm:s3://yangxiaoyu/librilight" \
                --embedding-dir $split_dir \
                --embedding-layer -1 \
                --quantizer-path $quantizer_path \
                --max-duration 250
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy small"
        pieces=$(find $split_dir -name "libriheavy_cuts_small.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_small.jsonl.gz
    fi
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Collect MVQ tokens on Libriheavy medium"
    
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
            python whisper/extract_whisper_mvq_client.py \
                --num-jobs 8 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --n-mels 128 \
                --embedding-dim 1280 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --s3-prefix "brainllm:s3://yangxiaoyu/librilight" \
                --embedding-dir $split_dir \
                --embedding-layer -1 \
                --quantizer-path $quantizer_path \
                --max-duration 250
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy medium"
        pieces=$(find $split_dir -name "libriheavy_cuts_medium.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_medium.jsonl.gz
    fi
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Collect MVQ tokens on Libriheavy large"
    
    subset=large
    num_splits=25
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data_s3/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy ${subset} split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python whisper/extract_whisper_mvq_client.py \
                --num-jobs 8 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --n-mels 128 \
                --embedding-dim 1280 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer -1 \
                --quantizer-path $quantizer_path \
                --max-duration 250
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy ${subset}"
        pieces=$(find $split_dir -name "libriheavy_cuts_${subset}.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_${subset}.jsonl.gz
    fi
fi


if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Stage 10: Collect MVQ tokens on wenetspeech L"
    
    subset=L
    num_splits=10
    split_dir=$vq_dir/wenetspeech_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split --no-pad $num_splits data/fbank_wenetspeech/wenetspeech_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding wenetspeech L split ${i}"
        if [ ! -f  $split_dir/wenetspeech_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python whisper/extract_whisper_mvq_client.py \
                --num-jobs 8 \
                --input-manifest $split_dir/wenetspeech_cuts_${subset}.0${i}.jsonl.gz \
                --target-manifest-file $split_dir/wenetspeech_cuts_${subset}.${i}.processed.jsonl.gz \
                --n-mels 128 \
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

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Stage 11: Collect MVQ tokens on WeneSpeech test/dev"
    for subset in DEV; do
        log "Processing $subset"
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 4 \
            --input-manifest data/fbank_wenetspeech/wenetspeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/wenetspeech_cuts_${subset}.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-wenetspeech-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

# TODO: add more datasets here

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    log "Stage 12: Collect MVQ tokens on various Chinese datasets"
    
    #for dataset in aidatatang_200zh aishell3 cs_wav acq; do
    #for dataset in aishell2 baidu_en_cn cantonese accent; do
    # for dataset in datatang1505 dialog dialog3k magicdata MagicData_dialog ximalaya zhvoice; do
    # for dataset in primewords_md_2018_set1 phone speech_wav; do
    # for dataset in speech_wav peoplespeech; do
    # for dataset in digital_library_202003 ST-CMDS-20170001_1-OS en_us_english en8848 ljspeech tatoeba ted vctk voase voaSplider; do
    for dataset in 20220309; do
        log "Processing $dataset"
        mkdir -p $vq_dir/${dataset}
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 8 \
            --input-manifest ASR_data/preprocessed_manifest/${dataset}_cuts.jsonl.gz \
            --target-manifest-file $vq_dir/${dataset}_cuts.jsonl.gz \
            --n-mels 128 \
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
    log "Stage 13: Collect MVQ tokens on MLS"
    num_splits=8
    split_dir=$vq_dir/MLS_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits ASR_data/preprocessed_manifest/MLS_cuts.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding MLS split ${i}"
        if [ ! -f  $split_dir/MLS_cuts.${i}.processed.jsonl.gz ]; then
            python whisper/extract_whisper_mvq_client.py \
                --num-jobs 8 \
                --input-manifest $split_dir/MLS_cuts.${i}.jsonl.gz \
                --target-manifest-file $split_dir/MLS_cuts.${i}.processed.jsonl.gz \
                --n-mels 128 \
                --embedding-dim 1280 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-MLS-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer -1 \
                --quantizer-path $quantizer_path \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/MLS_cuts.jsonl.gz ]; then
        log "Combining the processed cuts of MLS"
        pieces=$(find $split_dir -name "MLS_cuts.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/MLS_cuts.jsonl.gz
    fi
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
    log "Stage 14: Collect MVQ tokens on sensetime data"
    subset=sensetime
    num_splits=3
    split_dir=$vq_dir/${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits ASR_data/preprocessed_manifest/${subset}_cuts.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding ${subset} split ${i}"
        if [ ! -f  $split_dir/${subset}_cuts.${i}.processed.jsonl.gz ]; then
            python whisper/extract_whisper_mvq_client.py \
                --num-jobs 8 \
                --input-manifest $split_dir/${subset}_cuts.${i}.jsonl.gz \
                --target-manifest-file $split_dir/${subset}_cuts.${i}.processed.jsonl.gz \
                --n-mels 128 \
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
fi

if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
    log "Stage 15: Collect MVQ tokens on various English datasets"
    for dataset in common_voice_20200622 en_us_english; do
        log "Processing $dataset"
        mkdir -p $vq_dir/${dataset}
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 4 \
            --input-manifest ASR_data/preprocessed_manifest/${dataset}_cuts.jsonl.gz \
            --target-manifest-file $vq_dir/${dataset}_cuts.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-${dataset} \
            --embedding-dir $vq_dir/${dataset} \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 16 ] && [ $stop_stage -ge 16 ]; then
    log "Stage 16: Collect MVQ tokens on speech_annotations data"
    subset=speech_annotations_2021
    num_splits=10
    split_dir=$vq_dir/${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split --no-pad $num_splits ASR_data/preprocessed_manifest/${subset}_cuts.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    # for i in $(seq 0 1 $(($num_splits-1))); do
    # for i in $(seq 0 1 4); do
    for i in $(seq 5 1 $(($num_splits-1))); do
        log "Start encoding ${subset} split ${i}"
        if [ ! -f  $split_dir/${subset}_cuts.${i}.processed.jsonl.gz ]; then
            python whisper/extract_whisper_mvq_client.py \
                --num-jobs 8 \
                --input-manifest $split_dir/${subset}_cuts.${i}.jsonl.gz \
                --target-manifest-file $split_dir/${subset}_cuts.${i}.processed.jsonl.gz \
                --n-mels 128 \
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
fi

if [ $stage -le 17 ] && [ $stop_stage -ge 17 ]; then
    log "Stage 17: Collect MVQ tokens on wechat reading datasets"
    
    for dataset in weread-16k-res-00 weread-16k-res-01 weread-16k-res-02 weread-16k-res-03 weread-16k-res-04; do
        log "Processing $dataset"
        mkdir -p $vq_dir/${dataset}
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 8 \
            --input-manifest ASR_data/preprocessed_manifest/${dataset}_cuts.jsonl.gz \
            --target-manifest-file $vq_dir/${dataset}_cuts.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-${dataset} \
            --embedding-dir $vq_dir/${dataset} \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 18 ] && [ $stop_stage -ge 18 ]; then
    log "Stage 18: Collect MVQ tokens on alimeeting"
    dataset=alimeeting
    mkdir -p $vq_dir/${dataset}
    for subset in eval test; do
        log "Processing $subset"
        python whisper/extract_whisper_mvq_client.py \
            --num-jobs 1 \
            --input-manifest data/fbank_alimeeting/alimeeting-far_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/alimeeting-far_cuts_${subset}.jsonl.gz \
            --n-mels 128 \
            --embedding-dim 1280 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-alimeeting-${subset} \
            --embedding-dir $vq_dir/${dataset} \
            --embedding-layer -1 \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi


log "Done"