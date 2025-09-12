#!/usr/bin/env bash
export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/dasheng_dev:$PYTHONPATH

set -eou pipefail


stage=-1
stop_stage=-1

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

model_version=large
num_codebooks=8
embedding_layer=-1
normalize=0

# vq_dir=data/vq_dasheng_large_libri_as_cb_${num_codebooks}
vq_dir=data/vq_dasheng_large_layer_${embedding_layer}_normalize_${normalize}_cb_${num_codebooks}
mkdir -p $vq_dir
log "Manifest will be stored under: $vq_dir"

# quantizer_path=data/quantizer/dasheng-large-libri-as-cb-${num_codebooks}.pt
quantizer_path=data/quantizer/dasheng-large-layer-${embedding_layer}-normalize-${normalize}-cb-${num_codebooks}.pt
log "Using ${quantizer_path}"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python dasheng/train_mvq.py \
        --embedding-dim 1536 \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/dasheng/dasheng-layer--1-audioset-balanced.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/dasheng/dasheng-layer--1-audioset-eval.jsonl.gz
            
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on audioset sets"
    for subset in balanced eval; do
        python dasheng/extract_mvq.py \
            --num-jobs 2 \
            --model-version $model_version \
            --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on audioset sets"
    for subset in full; do
        python dasheng/extract_mvq.py \
            --num-jobs 6 \
            --model-version $model_version \
            --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer ${embedding_layer} \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Collect MVQ tokens on LibriSpeech training sets"
    for subset in dev-clean dev-other; do
        python dasheng/extract_mvq.py \
            --num-jobs 1 \
            --model-version $model_version \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer ${embedding_layer} \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
    
    
    for subset in train-all-shuf; do
        python dasheng/extract_mvq.py \
            --num-jobs 4 \
            --model-version $model_version \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer ${embedding_layer} \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done

fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Collect MVQ tokens on ESC"
    python dasheng/extract_mvq.py \
        --num-jobs 1 \
        --model-version $model_version \
        --input-manifest data/esc/esc_cuts.jsonl.gz \
        --target-manifest-file $vq_dir/esc_cuts.jsonl.gz \
        --embedding-dim 1536 \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-esc \
        --embedding-dir $vq_dir \
        --embedding-layer ${embedding_layer} \
        --quantizer-path $quantizer_path \
        --max-duration 200
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Collect MVQ tokens on Vox1-test"
    python dasheng/extract_mvq.py \
        --num-jobs 1 \
        --model-version $model_version \
        --input-manifest data/fbank_voxceleb/vox1_cuts_test.jsonl.gz \
        --target-manifest-file $vq_dir/vox1_test_cuts.jsonl.gz \
        --embedding-dim 1536 \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-vox1-test \
        --embedding-dir $vq_dir \
        --embedding-layer ${embedding_layer} \
        --quantizer-path $quantizer_path \
        --max-duration 200
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Collect MVQ tokens on music4all"
    python dasheng/extract_mvq.py \
        --num-jobs 8 \
        --model-version $model_version \
        --input-manifest data/music4all_manifest/music4all_cuts_all.jsonl.gz \
        --target-manifest-file $vq_dir/music4all_cuts_all.jsonl.gz \
        --embedding-dim 1536 \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-music4all-all \
        --embedding-dir $vq_dir \
        --embedding-layer ${embedding_layer} \
        --quantizer-path $quantizer_path \
        --max-duration 200
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Collect MVQ tokens on BBC dataset"
    for subset in train test; do
        python dasheng/extract_mvq.py \
            --num-jobs 8 \
            --model-version $model_version \
            --input-manifest data/bbc_soundeffect_manifest/bbc_soundeffect_cuts_${subset}_10s.jsonl.gz \
            --target-manifest-file $vq_dir/bbc_soundeffect_cuts_${subset}_10s.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-bbc-soundeffect-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Collect MVQ tokens on VGGsound"
    for subset in train test; do
        python dasheng/extract_mvq.py \
            --num-jobs 4 \
            --model-version $model_version \
            --input-manifest data/vggsound_manifest/vggsound_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/vggsound_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-vggsound-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Stage 10: Collect MVQ tokens on freesound"
    for subset in train_10s test_10s; do
        python dasheng/extract_mvq.py \
            --num-jobs 8 \
            --model-version $model_version \
            --input-manifest data/freesound_manifest/freesound_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/freesound_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-freesound-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Stage 11: Collect MVQ tokens on MTG"
    for subset in 10s; do
        embedding_dir=$vq_dir/mtg_wav
        mkdir -p $embedding_dir
        python dasheng/extract_mvq.py \
            --num-jobs 8 \
            --model-version $model_version \
            --input-manifest data/mtg_manifest_wav/mtg_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/mtg_wav_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-mtg-wav-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi


if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    log "Stage 12: Collect MVQ tokens on gigaspeech"
    for subset in dev xs s; do
        embedding_dir=$vq_dir/giga_${subset}
        mkdir -p $embedding_dir
        python dasheng/extract_mvq.py \
            --num-jobs 4 \
            --model-version $model_version \
            --input-manifest data/gigaspeech_manifest/gigaspeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-giga-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done

    for subset in m l xl; do
        embedding_dir=$vq_dir/giga_${subset}
        mkdir -p $embedding_dir
        python dasheng/extract_mvq.py \
            --num-jobs 8 \
            --model-version $model_version \
            --input-manifest data/gigaspeech_manifest/gigaspeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz \
            --embedding-dim 1536 \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-giga-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --normalize $normalize \
            --quantizer-path $quantizer_path \
            --max-duration 200
    done
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
    log "Stage 13: Collect MVQ tokens on libriheavy"
    
    subset=small
    num_splits=4
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data/fbank_libriheavy_split/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi

    for i in $(seq 0 1 $(($num_splits-1))); do
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python dasheng/extract_mvq.py \
                --num-jobs 8 \
                --model-version $model_version \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --embedding-dim 1536 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-${subset}-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --normalize $normalize \
                --quantizer-path $quantizer_path \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy $subset"
        pieces=$(find $split_dir -name "libriheavy_cuts_${subset}.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_${subset}.jsonl.gz
    fi
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
    log "Stage 14: Collect MVQ tokens on libriheavy"
    
    subset=medium
    num_splits=5
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data/fbank_libriheavy_split/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi

    for i in $(seq 0 1 $(($num_splits-1))); do
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python dasheng/extract_mvq.py \
                --num-jobs 8 \
                --model-version $model_version \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --embedding-dim 1536 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-${subset}-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --normalize $normalize \
                --quantizer-path $quantizer_path \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy $subset"
        pieces=$(find $split_dir -name "libriheavy_cuts_${subset}.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_${subset}.jsonl.gz
    fi
fi


if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
    log "Stage 15: Collect MVQ tokens on libriheavy"
    
    subset=large
    num_splits=20
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data/fbank_libriheavy_split/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi

    # for i in $(seq 0 1 6); do
    for i in $(seq 7 1 $(($num_splits-1))); do
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python dasheng/extract_mvq.py \
                --num-jobs 8 \
                --model-version $model_version \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --embedding-dim 1536 \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-${subset}-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --normalize $normalize \
                --quantizer-path $quantizer_path \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy $subset"
        pieces=$(find $split_dir -name "libriheavy_cuts_${subset}.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_${subset}.jsonl.gz
    fi
fi

if [ $stage -le 16 ] && [ $stop_stage -ge 16 ]; then
    log "Stage 7: Collect MVQ tokens on voxpopuli unlabelled english"
    for subset in en; do
        num_splits=8
        split_dir=$vq_dir/voxpopuli_${subset}_split
        mkdir -p $split_dir

        if [ ! -f $split_dir/.split_completed ]; then
            lhotse split $num_splits --no-pad data/voxpopuli_en_manifest/voxpopuli_cuts_${subset}.jsonl.gz $split_dir
            touch $split_dir/.split_completed
        fi

        for i in $(seq 0 1 $(($num_splits-1))); do
            log "Start encoding voxpopuli unlabelled ${subset} split ${i}"
            if [ ! -f $split_dir/voxpopuli_cuts_${subset}.${i}.processed.jsonl.gz ]; then
                python dasheng/extract_mvq.py \
                    --num-jobs 8 \
                    --model-version $model_version \
                    --input-manifest $split_dir/voxpopuli_cuts_${subset}.${i}.jsonl.gz \
                    --target-manifest-file $split_dir/voxpopuli_cuts_${subset}.${i}.processed.jsonl.gz \
                    --embedding-dim 1536 \
                    --num-codebooks $num_codebooks \
                    --manifest-name codebook-indexes-voxpopuli-en-${subset}-split-${i} \
                    --embedding-dir $split_dir \
                    --embedding-layer $embedding_layer \
                    --normalize $normalize \
                    --quantizer-path $quantizer_path \
                    --max-duration 200
            fi
        done
        if [ ! -f $vq_dir/voxpopuli_cuts_${subset}.jsonl.gz ]; then
            log "Combining the processed cuts of voxpopuli en"
            pieces=$(find $split_dir -name "voxpopuli_cuts_${subset}.*.processed.jsonl.gz")
            lhotse combine $pieces $vq_dir/voxpopuli_cuts_${subset}.jsonl.gz
        fi
    done
fi