#!/usr/bin/env bash
export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH

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

vq_dir=data/vq_whisper_turbo_zh_en_${num_codebooks}_v2
mkdir -p $vq_dir

quantizer_path=data/quantizer/whisper-turbo-zh-en-cb-${num_codebooks}-v2.pt
# quantizer_path=data/quantizer/whisper-turbo-libri-giga-cb-16.pt

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python local/train_mvq.py \
        --embedding-dim 1280 \
        --num-codebooks $num_codebooks \
        --feature-type h5 \
        --quantizer-path $quantizer_path \
        /mnt/workspace/xiaoyu/workspace/discrete_tokens/manifests/aishell_subset-whisper-turbo-layer--1.jsonl.gz \
        /mnt/workspace/xiaoyu/workspace/discrete_tokens/manifests/libri_giga_wenet_mix-whisper-turbo-layer--1.jsonl.gz
fi
        

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech dev/test sets"
    for subset in dev-clean dev-other test-clean test-other; do
        python multi_KD/extract_whisper_mvq.py \
            --num-jobs 1 \
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

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on LibriSpeech training sets"
    for subset in train-all-shuf; do
        python multi_KD/extract_whisper_mvq.py \
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
        python multi_KD/extract_whisper_mvq.py \
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
        python multi_KD/extract_whisper_mvq.py \
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
            python multi_KD/extract_whisper_mvq.py \
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
        lhotse split $num_splits data/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy small split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python multi_KD/extract_whisper_mvq.py \
                --num-jobs 4 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --n-mels 128 \
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

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Collect MVQ tokens on Libriheavy medium"
    
    subset=medium
    num_splits=5
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits data/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy medium split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python multi_KD/extract_whisper_mvq.py \
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
                --max-duration 200
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
        lhotse split $num_splits --no-pad data/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy ${subset} split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python multi_KD/extract_whisper_mvq.py \
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
        lhotse split $num_splits data/fbank_wenetspeech/wenetspeech_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding wenetspeech L split ${i}"
        if [ ! -f  $split_dir/wenetspeech_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python multi_KD/extract_whisper_mvq.py \
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
    log "Stage 10: Collect MVQ tokens on WeneSpeech test/dev"
    for subset in DEV; do
        log "Processing $subset"
        python multi_KD/extract_whisper_mvq.py \
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


log "Done"