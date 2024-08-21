#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=0
stop_stage=100

. shared/parse_options.sh || exit 1

dl_dir=download
mkdir -p $dl_dir

manifest_dir=data/fbank_multi_KD
mkdir -p $manifest_dir

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "manifest_dir: $manifest_dir"

log "We assume you have finished the data preparation of librispeech "
log "by running: ./prepare.sh --stage 0 --stop_stage 3"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Download Audioset from https://huggingface.co/datasets/marcoyang/audioset-full/tree/main to dl_dir"

    # combine the splits
    if [ ! -e $dl_dir/.audioset.done ]; then
        cat download/audioset.tar.gz.part* > $dl_dir/audioset.tar.gz
        tar -xvzf download/audioset.tar.gz -C $dl_dir
        touch $dl_dir/.audioset.done
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Compute fbank for audioset and generate manifest"
    if [ ! -e data/fbank_audioset/.audioset.done ]; then
        for split in balanced unbalanced eval; do
            python ./local/generate_audioset_manifest.py \
                --dataset-dir download/audioset \
                --split $split \
                --feat-output-dir data/fbank_audioset
        done
        touch data/fbank_audioset/.audioset.done
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Compute fbank for VoxCeleb and generate manifest"
    log "You need to first download VoxCeleb1 and VoxCeleb2 to $dl_dir"
    log "After downloading, convert the w4a to wav: https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830"
    
    if [ ! -e data/fbank/.voxceleb.done ]; then
        for dataset in vox1 vox2; do
            for part in dev test; do
                python ./local/generate_voxceleb_manifest.py \
                    --dataset $dataset \
                    --part $part \
                    --manifest-output-dir data/fbank/cuts_${dataset}_${part}.jsonl.gz
            done
        done
        touch data/fbank/.voxceleb.done
    fi

    log "Download the evaluate pairs"
    wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt -P $dl_dir
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect Whisper embeddings and create manifest accordingly"
    for key in train-all-shuf dev-clean dev-other; do
        for embedding_layer in -1; do
            python multi_KD/collect_whisper_embeddings.py \
                --num-jobs 4 \
                --input-manifest data/fbank/librispeech_cuts_${key}.jsonl.gz \
                --manifest-name embeddings-${key} \
                --target-manifest-file ${manifest_dir}/librispeech_cuts_${key}.jsonl.gz \
                --embedding-layer $embedding_layer \
                --max-duration 300 \
                --whisper-version large-v3
        done
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Collect BEATs embeddings and create manifest accordingly"
    
    if [ ! -e data/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt ]; then
        log "Download BEATs model"
        mkdir -p data/models
        wget https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea -P data/models
    fi
    
    for part in balanced unbalanced eval; do
        python multi_KD/collect_beats_embeddings.py \
            --num-jobs 4 \
            --input-manifest data/fbank_audioset/cuts_audioset_${part}.jsonl.gz \
            --manifest-name embeddings-$part \
            --max-duration 1000 \
            --target-manifest-file ${manifest_dir}/cuts_audioset_${part}.jsonl.gz
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Collect ECAPA-TDNN embeddings for VoxCeleb and create manifest accordingly"
    for part in vox1_test vox1_dev vox2_test vox2_dev; do
        python multi_KD/collect_ecapa_embeddings.py \
            --num-jobs 4 \
            --input-manifest data/fbank/cuts_${part}.jsonl.gz \
            --manifest-name embeddings-$part \
            --target-manifest-file ${manifest_dir}/cuts_${part}.jsonl.gz \
            --max-duration 1000
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Collect ECAPA-TDNN embeddings for LibriSpeech and create manifest accordingly"
    log "This is necessary if you share ASR data for SV"
    log "This is recommended!"
    for part in train-all-shuf dev-other dev-clean; do
        python multi_KD/collect_ecapa_embeddings.py \
            --num-jobs 4 \
            --input-manifest ${manifest_dir}/librispeech_cuts_${part}.jsonl.gz \
            --manifest-name embeddings-$part \
            --target-manifest-file ${manifest_dir}/librispeech_cuts_${part}-with-speaker-embed.jsonl.gz \
            --max-duration 1000
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Create dummy embeddings"
    log "This is for the convenience of data loading, this ensures that"
    log "Every cut has teacher embeddings from every teacher (though fake)"
    
    mkdir -p data/dummy_embeddings
    python local/prepare_dummy_embeddings.py 
fi

# The following are for FMA dataset (optional)
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Download FMA dataset"
    log "Please first follow the instructions at https://github.com/mdeff/fma"
    log "Then make a softlink under $dl_dir via: ln -svf FMA_DIR fma"

    if [ ! -e data/fbank/.fma.done ]; then
        for part in large; do
            python ./local/generate_fma_manifest.py \
                --dataset-dir download/fma \
                --split $part \
                --extension mp3
        done
        touch data/fbank/.fma.done
    fi
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Extract MERT embeddings on FMA"
    for part in large; do
        python multi_KD/collect_mert_embeddings.py \
            --num-jobs 4 \
            --input-manifest data/fbank_fma/cuts_fma_${part}.jsonl.gz \
            --embedding-level frame \
            --output-manifest embeddings-fma-${part} \
            --mert-version MERT-v1-330M \
            --max-duration 500
    done
fi

# The following are for Gigaspeech dataset (optional)
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Stage 10: Download gigaspeech"
    log "Please run ./prepare_gigaspeech.sh --stage 0 --stop_stage 11 \
    to preprocess gigaspeech dataset"
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Stage 11: Extract Whisper embedding on Gigaspeech"
    log "This assume that you already downloaded Gigaspeech and prepared fbank for gigaspeech"
    for key in XL DEV TEST; do
        python multi_KD/collect_whisper_embeddings.py \
            --num-jobs 1 \
            --input-manifest data/fbank/gigaspeech_cuts_${key}.jsonl.gz \
            --manifest-name embeddings-${key} \
            --target-manifest-file ${manifest_dir}/gigaspeech_cuts_${key}.jsonl.gz \
            --embedding-layer -1 \
            --max-duration 300 \
            --whisper-version large-v3
    done
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    log "Stage 6: Collect ECAPA-TDNN embeddings for GigaSpeech and create manifest accordingly"
    log "This is necessary if you share ASR data for SV"
    for part in XL DEV TEST; do
        python multi_KD/collect_ecapa_embeddings.py \
            --num-jobs 4 \
            --input-manifest ${manifest_dir}/gigaspeech_cuts_${part}.jsonl.gz \
            --manifest-name embeddings-$part \
            --target-manifest-file ${manifest_dir}/gigaspeech_cuts_${part}-with-speaker-embed.jsonl.gz \
            --max-duration 1000
    done
fi