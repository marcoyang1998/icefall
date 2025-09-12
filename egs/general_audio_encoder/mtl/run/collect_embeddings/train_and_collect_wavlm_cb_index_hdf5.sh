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

# vq_dir=data_hdf5/vq_wavlm_${model_version}_layer_${embedding_layer}_normalize_${normalize}_libri_cb_${num_codebooks}
vq_dir=data_hdf5/vq_wavlm_cb${num_codebooks}_dasheng_cb8_combined
mkdir -p $vq_dir

quantizer_path=data/quantizer/wavlm-${model_version}-layer-${embedding_layer}-normalize-${normalize}-libri-cb-${num_codebooks}.pt

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python local/train_mvq.py \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/wavlm/wavlm-${model_version}-layer-${embedding_layer}-sampled.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/wavlm/wavlm-large-layer-${embedding_layer}-dev-clean.jsonl.gz \
            data/manifests/wavlm/wavlm-large-layer-${embedding_layer}-dev-other.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech training sets"
    for subset in train-all-shuf train-clean-100; do
        embedding_dir=$vq_dir/librispeech_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done

    for subset in dev-clean dev-other; do
        embedding_dir=$vq_dir/librispeech_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 2 \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-libri-${subset} \
            --embedding-dir $embedding_dir \
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
        lhotse split $num_splits --no-pad data/fbank_libriheavy_split/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy small split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python wavlm/extract_mvq_hdf5.py \
                --num-jobs 8 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --wavlm-version $model_version \
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
        lhotse split $num_splits --no-pad data/fbank_libriheavy_split/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy ${subset} split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python wavlm/extract_mvq_hdf5.py \
                --num-jobs 8 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --wavlm-version $model_version \
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
        lhotse split $num_splits --no-pad data/fbank_libriheavy_split/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    
    for i in $(seq 0 1 $(($num_splits-1))); do
        log "Start encoding libriheavy ${subset} split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python wavlm/extract_mvq_hdf5.py \
                --num-jobs 8 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --wavlm-version $model_version \
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
    giga_manifest=data_hdf5/vq_wavlm_large_layer_21_normalize_1_cb_16
    for subset in dev xs s; do
        embedding_dir=$vq_dir/giga_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 4 \
            --input-manifest $giga_manifest/gigaspeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-giga-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done

    for subset in m l xl; do
        embedding_dir=$vq_dir/giga_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --input-manifest $giga_manifest/gigaspeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/gigaspeech_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-giga-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
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
                python wavlm/extract_mvq_hdf5.py \
                    --num-jobs 8 \
                    --input-manifest $split_dir/voxpopuli_cuts_${subset}.${i}.jsonl.gz \
                    --target-manifest-file $split_dir/voxpopuli_cuts_${subset}.${i}.processed.jsonl.gz \
                    --wavlm-version $model_version \
                    --embedding-dim $model_dim \
                    --num-codebooks $num_codebooks \
                    --manifest-name codebook-indexes-voxpopuli-en-${subset}-split-${i} \
                    --embedding-dir $split_dir \
                    --embedding-layer $embedding_layer \
                    --quantizer-path $quantizer_path \
                    --normalize $normalize \
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

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Collect MVQ tokens on msp-podcast"
    for subset in Train Test1 Test2 Development; do
        embedding_dir=$vq_dir/msp_podcast
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --input-manifest data/msp_podcast_manifest/msp_podcast_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/msp_podcast_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-msp-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Stage 10: Collect MVQ tokens on iemocap all"

    for subset in all; do
        embedding_dir=$vq_dir/iemocap
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 1 \
            --input-manifest data/iemocap_manifest/iemocap_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/iemocap_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-iemocap-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Stage 11: Collect MVQ tokens on MELD"

    for subset in train dev test; do
        embedding_dir=$vq_dir/meld
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 2 \
            --input-manifest data/meld_manifest/meld_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/meld_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-meld-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    log "Stage 12: Collect MVQ tokens on MEAD"

    for subset in all; do
        embedding_dir=$vq_dir/mead
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 2 \
            --input-manifest data/mead_manifest/mead_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/mead_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
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

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
    log "Stage 13: Collect MVQ tokens on Fisher"

    for subset in part1 part2; do
        embedding_dir=$vq_dir/fisher_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --input-manifest data/fisher_manifest/fisher_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/fisher_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-fisher-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
    log "Stage 14: Collect MVQ tokens on voxpopuli"

    for lang in en de fr es pl it ro hu cs nl fi hr sk sl et lt; do
        for subset in train; do
            embedding_dir=$vq_dir/voxpopuli_${lang}
            mkdir -p $embedding_dir
            python wavlm/extract_mvq_hdf5.py \
                --num-jobs 4 \
                --input-manifest data_hdf5/voxpopuli_manifest_trimmed/voxpopuli-asr-${lang}_cuts_${subset}.jsonl.gz \
                --target-manifest-file $vq_dir/voxpopuli-asr-${lang}_cuts_${subset}.jsonl.gz \
                --wavlm-version $model_version \
                --embedding-dim $model_dim \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-voxpopuli-${lang}-${subset} \
                --embedding-dir $embedding_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 200
        done

        for subset in dev test; do
            python wavlm/extract_mvq_hdf5.py \
                --num-jobs 1 \
                --input-manifest data_hdf5/voxpopuli_manifest_trimmed/voxpopuli-asr-${lang}_cuts_${subset}.jsonl.gz \
                --target-manifest-file $vq_dir/voxpopuli-asr-${lang}_cuts_${subset}.jsonl.gz \
                --wavlm-version $model_version \
                --embedding-dim $model_dim \
                --num-codebooks $num_codebooks \
                --manifest-name codebook-indexes-voxpopuli-${lang}-${subset} \
                --embedding-dir $embedding_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 200
        done
    done
fi

if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
    log "Stage 15: Collect MVQ tokens on common voice 17 english"
    for subset in train; do
        embedding_dir=$vq_dir/commonvoice17_en_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --input-manifest data/cv17_manifest/commonvoice_en_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/commonvoice_en_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-cv17-en-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done

    for subset in dev test; do
        embedding_dir=$vq_dir/commonvoice17_en_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 2 \
            --input-manifest data/cv17_manifest/commonvoice_en_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/commonvoice_en_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-cv17-en-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 16 ] && [ $stop_stage -ge 16 ]; then
    log "Stage 16: Collect MVQ tokens on spgispeech"
    for subset in train; do
        embedding_dir=$vq_dir/spgispeech_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 8 \
            --input-manifest data/spgispeech_manifest/spgispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/spgispeech_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-spgispeech-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done

    for subset in dev test; do
        embedding_dir=$vq_dir/spgispeech_${subset}
        mkdir -p $embedding_dir
        python wavlm/extract_mvq_hdf5.py \
            --num-jobs 4 \
            --input-manifest data/spgispeech_manifest/spgispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/spgispeech_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-spgispeech-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 17 ] && [ $stop_stage -ge 17 ]; then
    log "Stage 17: Collect MVQ tokens audioset"

    embedding_dir=$vq_dir/audioset
    mkdir -p $embedding_dir
    for subset in balanced eval full; do
        python wavlm/extract_mvq_hdf5_and_combine.py \
            --num-jobs 8 \
            --input-manifest data/vq_dasheng_large_layer_-1_normalize_0_cb_8/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-audioset-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 18 ] && [ $stop_stage -ge 18 ]; then
    log "Stage 18: Collect MVQ tokens music4all"

    embedding_dir=$vq_dir/music4all
    mkdir -p $embedding_dir
    
    python wavlm/extract_mvq_hdf5_and_combine.py \
        --num-jobs 8 \
        --input-manifest data/vq_dasheng_large_layer_-1_normalize_0_cb_8/music4all_cuts_all.jsonl.gz \
        --target-manifest-file $vq_dir/music4all_cuts_all.jsonl.gz \
        --wavlm-version $model_version \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-music4all-all \
        --embedding-dir $embedding_dir \
        --embedding-layer $embedding_layer \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --max-duration 200
fi

if [ $stage -le 19 ] && [ $stop_stage -ge 19 ]; then
    log "Stage 19: Collect MVQ tokens VGGsound"

    embedding_dir=$vq_dir/vggsound
    mkdir -p $embedding_dir
    
    for subset in train test; do
        python wavlm/extract_mvq_hdf5_and_combine.py \
            --num-jobs 4 \
            --input-manifest data/vq_dasheng_large_layer_-1_normalize_0_cb_8/vggsound_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/vggsound_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-vggsound-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    log "Stage 20: Collect MVQ tokens freesound"

    embedding_dir=$vq_dir/freesound
    mkdir -p $embedding_dir
    
    for subset in train_10s test_10s; do
        python wavlm/extract_mvq_hdf5_and_combine.py \
            --num-jobs 8 \
            --input-manifest data/vq_dasheng_large_layer_-1_normalize_0_cb_8/freesound_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/freesound_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-freesound-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]; then
    log "Stage 21: Collect MVQ tokens on MTG"

    embedding_dir=$vq_dir/mtg_wav
    mkdir -p $embedding_dir
    
    for subset in 10s; do
        python wavlm/extract_mvq_hdf5_and_combine.py \
            --num-jobs 8 \
            --input-manifest data/vq_dasheng_large_layer_-1_normalize_0_cb_8/mtg_wav_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/mtg_wav_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-mtg-wav-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi

if [ $stage -le 22 ] && [ $stop_stage -ge 22 ]; then
    log "Stage 22: Collect MVQ tokens on BBC soundeffect"

    embedding_dir=$vq_dir/bbc_soundeffect
    mkdir -p $embedding_dir
    
    for subset in train_10s test_10s; do
        python wavlm/extract_mvq_hdf5_and_combine.py \
            --num-jobs 8 \ 
            --input-manifest data/vq_dasheng_large_layer_-1_normalize_0_cb_8/bbc_soundeffect_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/bbc_soundeffect_cuts_${subset}.jsonl.gz \
            --wavlm-version $model_version \
            --embedding-dim $model_dim \
            --num-codebooks $num_codebooks \
            --manifest-name codebook-indexes-bbc-soundeffect-${subset} \
            --embedding-dir $embedding_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 200
    done
fi


if [ $stage -le 101 ] && [ $stop_stage -ge 101 ]; then
    log "Stage 101: Collect MVQ tokens voxceleb"

    embedding_dir=$vq_dir/vox1
    mkdir -p $embedding_dir
    python wavlm/extract_mvq_hdf5.py \
        --num-jobs 1 \
        --input-manifest data/fbank_voxceleb/vox1_cuts_test.jsonl.gz \
        --target-manifest-file $vq_dir/vox1_cuts_test.jsonl.gz \
        --wavlm-version $model_version \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-vox1-test \
        --embedding-dir $embedding_dir \
        --embedding-layer $embedding_layer \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --max-duration 200
fi


# if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
#     log "Stage 30: Collect MVQ tokens on LibriSpeech training sets"
#     for subset in small; do
#         python wavlm/extract_mvq_hdf5.py \
#             --num-jobs 1 \
#             --input-manifest data/fbank_libriheavy_split/libriheavy_cuts_${subset}.jsonl.gz \
#             --target-manifest-file $vq_dir/libriheavy_cuts_${subset}.jsonl.gz \
#             --wavlm-version $model_version \
#             --embedding-dim $model_dim \
#             --num-codebooks $num_codebooks \
#             --manifest-name codebook-indexes-lh-${subset} \
#             --s3-prefix $prefix_folder/librilight/${subset} \
#             --embedding-dir $vq_dir \
#             --embedding-layer $embedding_layer \
#             --quantizer-path $quantizer_path \
#             --normalize $normalize \
#             --max-duration 200
#     done
# fi

# if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
#     for subset in dev-clean dev-other; do
#         python wavlm/extract_mvq_hdf5.py \
#             --num-jobs 1 \
#             --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
#             --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
#             --wavlm-version $model_version \
#             --embedding-dim $model_dim \
#             --num-codebooks $num_codebooks \
#             --manifest-name codebook-indexes-libri-${subset} \
#             --s3-prefix "/cpfs02/user/housiyuan/xiaoyu/codebook_indexes/wavlm_${model_version}_layer_${embedding_layer}" \
#             --embedding-dir $vq_dir \
#             --embedding-layer $embedding_layer \
#             --quantizer-path $quantizer_path \
#             --normalize $normalize \
#             --max-duration 200
#     done
# fi