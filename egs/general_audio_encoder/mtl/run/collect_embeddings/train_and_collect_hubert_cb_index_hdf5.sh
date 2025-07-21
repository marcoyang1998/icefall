#!/usr/bin/env bash
export PYTHONPATH=./../../../:$PYTHONPATH

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
model_dim=1024
embedding_layer=21
num_codebooks=16
normalize=1

vq_dir=data_hdf5/vq_hubert_${model_version}_layer_${embedding_layer}_normalize_${normalize}_cb_${num_codebooks}
mkdir -p $vq_dir

quantizer_path=data/quantizer/hubert-${model_version}-layer-${embedding_layer}-normalize-${normalize}-cb-${num_codebooks}.pt

log "Saving the codebook indexes with HDF5"
log "VQ dir: $vq_dir"
log "Quantizer: $quantizer_path"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python local/train_mvq.py \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/hubert/hubert-${model_version}-layer-${embedding_layer}-giga-sampled.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/hubert/hubert-large-layer-${embedding_layer}-giga-dev.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech training sets"
    for subset in train-all-shuf; do
        python hubert/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --hubert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done

    for subset in dev-clean dev-other; do
        python hubert/extract_mvq_hdf5.py \
            --num-jobs 2 \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --hubert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on Libriheavy small"
    subset=small
    num_splits=4
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy small split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python hubert/extract_mvq_hdf5.py \
                --num-jobs 4 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --hubert-version $model_version \
                --embedding-dim $model_dim \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 250
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy small"
        pieces=$(find $split_dir -name "libriheavy_cuts_small.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_small.jsonl.gz
    fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Collect MVQ tokens on Libriheavy medium"
    
    subset=medium
    num_splits=5
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy ${subset} split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python hubert/extract_mvq_hdf5.py \
                --num-jobs 7 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --hubert-version $model_version \
                --embedding-dim $model_dim \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 250
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy ${subset}"
        pieces=$(find $split_dir -name "libriheavy_cuts_${subset}.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_${subset}.jsonl.gz
    fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Collect MVQ tokens on Libriheavy large"
    
    subset=large
    num_splits=20
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 12 1 $(($num_splits-1))); do
        log "Start encoding libriheavy ${subset} split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python hubert/extract_mvq_hdf5.py \
                --num-jobs 7 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --hubert-version $model_version \
                --embedding-dim $model_dim \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 250
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy ${subset}"
        pieces=$(find $split_dir -name "libriheavy_cuts_${subset}.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_${subset}.jsonl.gz
    fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Collect MVQ tokens on gigaspeech training sets"
    for subset in dev xs s; do
        python hubert/extract_mvq_hdf5.py \
            --num-jobs 2 \
            --input-manifest data/fbank_gigaspeech/gigaspeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz \
            --hubert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-giga-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 60 ] && [ $stop_stage -ge 60 ]; then
    log "Stage 60: Collect MVQ tokens on gigaspeech training sets"
    for subset in m l; do
        python hubert/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --input-manifest data/fbank_gigaspeech/gigaspeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz \
            --hubert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-giga-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Collect MVQ tokens on GigaSpeech xl"
    
    subset=xl
    num_splits=6
    split_dir=$vq_dir/giga_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits data/fbank_gigaspeech/gigaspeech_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding giga xl split ${i}"
        if [ ! -f  $split_dir/gigaspeech_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python hubert/extract_mvq_hdf5.py \
                --num-jobs 8 \
                --input-manifest $split_dir/gigaspeech_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/gigaspeech_cuts_${subset}.${i}.processed.jsonl.gz \
                --hubert-version $model_version \
                --embedding-dim $model_dim \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-giga-$subset-split-${i} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 200
        fi
    done

    if [ ! -f $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of gigaspeech xl"
        pieces=$(find $split_dir -name "gigaspeech_cuts_xl.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/gigaspeech_cuts_xl.jsonl.gz
    fi
fi

if [ $stage -le 80 ] && [ $stop_stage -ge 80 ]; then
    log "Stage 80: Collect MVQ tokens on esc training sets"
    python hubert/extract_mvq_hdf5.py \
        --num-jobs 4 \
        --input-manifest data/esc/esc_cuts.jsonl.gz \
        --target-manifest-file $vq_dir/esc_cuts.jsonl.gz \
        --hubert-version $model_version \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-esc \
        --embedding-dir $vq_dir \
        --embedding-layer $embedding_layer \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --max-duration 200
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Collect MVQ tokens on vox1 test training sets"
    python hubert/extract_mvq.py \
        --num-jobs 1 \
        --input-manifest data/fbank_voxceleb/vox1_cuts_test.jsonl.gz \
        --target-manifest-file $vq_dir/vox1_test_cuts.jsonl.gz \
        --hubert-version $model_version \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-vox1-test \
        --embedding-dir $vq_dir \
        --embedding-layer $embedding_layer \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --max-duration 200
fi


if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Collect MVQ tokens on msp-podcast"

    for subset in Train Test1 Test2 Development; do
        python hubert/extract_mvq.py \
            --num-jobs 2 \
            --input-manifest data/msp_podcast_manifest/msp_podcast_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/msp_podcast_cuts_${subset}.jsonl.gz \
            --hubert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-msp-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Stage 10: Collect MVQ tokens on iemocap all"

    for subset in all; do
        python hubert/extract_mvq.py \
            --num-jobs 1 \
            --input-manifest data/iemocap_manifest/iemocap_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/iemocap_cuts_${subset}.jsonl.gz \
            --hubert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-iemocap-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Stage 11: Collect MVQ tokens on MELD"

    for subset in train dev test; do
        python hubert/extract_mvq.py \
            --num-jobs 1 \
            --input-manifest data/meld_manifest/meld_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/meld_cuts_${subset}.jsonl.gz \
            --hubert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-meld-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    log "Stage 12: Collect MVQ tokens on MEAD"

    for subset in all; do
        python hubert/extract_mvq.py \
            --num-jobs 1 \
            --input-manifest data/mead_manifest/mead_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/mead_cuts_${subset}.jsonl.gz \
            --hubert-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-mead-${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi


# if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
#     log "Stage 30: Collect MVQ tokens on LibriSpeech training sets"
#     for subset in small; do
#         python hubert/extract_mvq.py \
#             --num-jobs 1 \
#             --input-manifest data/fbank_libriheavy/libriheavy_cuts_${subset}.jsonl.gz \
#             --target-manifest-file $vq_dir/libriheavy_cuts_${subset}.jsonl.gz \
#             --hubert-version $model_version \
#             --embedding-dim $model_dim \
#             --num-codebooks $num_codebooks \
#             --manifest-name codebook-indexes-lh-${subset} \
#             --embedding-dir $vq_dir \
#             --embedding-layer $embedding_layer \
#             --quantizer-path $quantizer_path \
#             --normalize $normalize \
#             --max-duration 200
#     done
# fi

# if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
#     for subset in dev-clean dev-other; do
#         python hubert/extract_mvq.py \
#             --num-jobs 1 \
#             --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
#             --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
#             --hubert-version $model_version \
#             --embedding-dim $model_dim \
#             --num-codebooks $num_codebooks \
#             --manifest-name codebook-indexes-libri-${subset} \
#             --embedding-dir $vq_dir \
#             --embedding-layer $embedding_layer \
#             --quantizer-path $quantizer_path \
#             --normalize $normalize \
#             --max-duration 200
#     done
# fi