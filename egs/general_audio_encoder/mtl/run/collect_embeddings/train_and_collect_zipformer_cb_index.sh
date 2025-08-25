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

model_name=zipformer
model_version=300m-lh-as-multi-mvq-pretrained
model_dim=1024
embedding_layer=-1
num_codebooks=16
normalize=0

vq_dir=data/vq_zipformer_${model_version}_layer_${embedding_layer}_normalize_${normalize}_cb_${num_codebooks}
mkdir -p $vq_dir

model_ckpt=zipformer_audio_encoder/exp-300m-zipformer-lh-large-as-full-multi-mvq-hubert-large-mvq-cb16-1.0-dasheng-as-mvq-cb8-0.2-mask-ratio-1.0-musan-1-larger-mask-rerun/iter-500000-avg-4.pt
quantizer_path=data/quantizer/zipformer-${model_version}-layer-${embedding_layer}-normalize-${normalize}-cb-${num_codebooks}.pt
prefix_folder=/cpfs02/user/housiyuan/xiaoyu/codebook_indexes/zipformer_${model_version}_layer_${embedding_layer}_normalize_${normalize}_cb_${num_codebooks}

log "VQ dir: $vq_dir"
log "Quantizer: $quantizer_path"
log "Prefix: $prefix_folder"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Train the quantizer"
    python local/train_mvq.py \
        --embedding-dim $model_dim \
        --num-codebooks $num_codebooks \
        --quantizer-path $quantizer_path \
        --normalize $normalize \
        --quantizer-training-manifests \
            data/manifests/zipformer/zipformer-300m-lh-as-multi-mvq-pretrained-layer--1-audioset-balanced.jsonl.gz \
            data/manifests/zipformer/zipformer-300m-lh-as-multi-mvq-pretrained-layer--1-sampled.jsonl.gz \
        --quantizer-evaluation-manifests \
            data/manifests/zipformer/zipformer-300m-lh-as-multi-mvq-pretrained-layer--1-dev-clean.jsonl.gz \
            data/manifests/zipformer/zipformer-300m-lh-as-multi-mvq-pretrained-layer--1-dev-other.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Collect MVQ tokens on LibriSpeech training sets"
    for subset in dev-clean dev-other; do
        python zipformer_audio_encoder/extract_mvq.py \
            --num-jobs 1 \
            --input-manifest  data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --zipformer-version $model_version \
            --model-ckpt $model_ckpt \
            --embedding-dim $model_dim \
            --num-cb $num_codebooks \
            --num-encoder-layers 2,2,4,5,4,2 \
            --feedforward-dim 512,1024,2048,3072,2048,1024 \
            --encoder-dim 192,384,768,1024,768,384 \
            --encoder-unmasked-dim 192,256,320,512,320,256 \
            --manifest-name codebook-indexes-libri-${subset} \
            --s3-prefix $prefix_folder/LibriSpeech/${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 400
    done

    for subset in train-all-shuf; do
        python zipformer_audio_encoder/extract_mvq.py \
            --num-jobs 4 \
            --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/librispeech_cuts_${subset}.jsonl.gz \
            --zipformer-version $model_version \
            --model-ckpt $model_ckpt \
            --embedding-dim $model_dim \
            --num-cb $num_codebooks \
            --num-encoder-layers 2,2,4,5,4,2 \
            --feedforward-dim 512,1024,2048,3072,2048,1024 \
            --encoder-dim 192,384,768,1024,768,384 \
            --encoder-unmasked-dim 192,256,320,512,320,256 \
            --manifest-name codebook-indexes-libri-${subset} \
            --s3-prefix $prefix_folder/LibriSpeech/${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 400
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Collect MVQ tokens on audioset training sets"

    for subset in balanced eval full; do
        python zipformer_audio_encoder/extract_mvq.py \
            --num-jobs 4 \
            --input-manifest  data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
            --target-manifest-file $vq_dir/audioset_cuts_${subset}.jsonl.gz \
            --zipformer-version $model_version \
            --model-ckpt $model_ckpt \
            --embedding-dim $model_dim \
            --num-cb $num_codebooks \
            --num-encoder-layers 2,2,4,5,4,2 \
            --feedforward-dim 512,1024,2048,3072,2048,1024 \
            --encoder-dim 192,384,768,1024,768,384 \
            --encoder-unmasked-dim 192,256,320,512,320,256 \
            --manifest-name codebook-indexes-audioset-${subset} \
            --s3-prefix $prefix_folder/audioset/${subset} \
            --embedding-dir $vq_dir \
            --embedding-layer $embedding_layer \
            --quantizer-path $quantizer_path \
            --normalize $normalize \
            --max-duration 400
    done

fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Collect MVQ tokens on Libriheavy small"
    
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
            python zipformer_audio_encoder/extract_mvq.py \
                --num-jobs 4 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --zipformer-version $model_version \
                --model-ckpt $model_ckpt \
                --embedding-dim $model_dim \
                --num-cb $num_codebooks \
                --num-encoder-layers 2,2,4,5,4,2 \
                --feedforward-dim 512,1024,2048,3072,2048,1024 \
                --encoder-dim 192,384,768,1024,768,384 \
                --encoder-unmasked-dim 192,256,320,512,320,256 \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --s3-prefix $prefix_folder/librilight/${subset} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 400
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy small"
        pieces=$(find $split_dir -name "libriheavy_cuts_small.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_small.jsonl.gz
    fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Collect MVQ tokens on Libriheavy medium"
    
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
            python zipformer_audio_encoder/extract_mvq.py \
                --num-jobs 8 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --zipformer-version $model_version \
                --model-ckpt $model_ckpt \
                --embedding-dim $model_dim \
                --num-cb $num_codebooks \
                --num-encoder-layers 2,2,4,5,4,2 \
                --feedforward-dim 512,1024,2048,3072,2048,1024 \
                --encoder-dim 192,384,768,1024,768,384 \
                --encoder-unmasked-dim 192,256,320,512,320,256 \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --s3-prefix $prefix_folder/librilight/${subset} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 400
        fi
    done

    if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Combining the processed cuts of libriheavy ${subset}"
        pieces=$(find $split_dir -name "libriheavy_cuts_${subset}.*.processed.jsonl.gz")
        lhotse combine $pieces $vq_dir/libriheavy_cuts_${subset}.jsonl.gz
    fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Collect MVQ tokens on Libriheavy large"
    
    subset=large
    num_splits=20
    split_dir=$vq_dir/libriheavy_${subset}_split
    mkdir -p $split_dir

    if [ ! -f $split_dir/.split_completed ]; then
        lhotse split $num_splits --no-pad data/fbank_libriheavy_split/libriheavy_cuts_${subset}.jsonl.gz $split_dir
        touch $split_dir/.split_completed
    fi
    
    for i in $(seq 11 1 $(($num_splits-1))); do
    # for i in $(seq 0 1 10); do
        log "Start encoding libriheavy ${subset} split ${i}"
        if [ ! -f $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz ]; then
            python zipformer_audio_encoder/extract_mvq.py \
                --num-jobs 8 \
                --input-manifest $split_dir/libriheavy_cuts_${subset}.${i}.jsonl.gz \
                --target-manifest-file $split_dir/libriheavy_cuts_${subset}.${i}.processed.jsonl.gz \
                --zipformer-version $model_version \
                --model-ckpt $model_ckpt \
                --embedding-dim $model_dim \
                --num-cb $num_codebooks \
                --num-encoder-layers 2,2,4,5,4,2 \
                --feedforward-dim 512,1024,2048,3072,2048,1024 \
                --encoder-dim 192,384,768,1024,768,384 \
                --encoder-unmasked-dim 192,256,320,512,320,256 \
                --manifest-name codebook-indexes-lh-$subset-split-${i} \
                --s3-prefix $prefix_folder/librilight/${subset} \
                --embedding-dir $split_dir \
                --embedding-layer $embedding_layer \
                --quantizer-path $quantizer_path \
                --normalize $normalize \
                --max-duration 400
        fi
    done

    # if [ ! -f $vq_dir/libriheavy_cuts_${subset}.jsonl.gz ]; then
    #     log "Combining the processed cuts of libriheavy ${subset}"
    #     pieces=$(find $split_dir -name "libriheavy_cuts_${subset}.*.processed.jsonl.gz")
    #     lhotse combine $pieces $vq_dir/libriheavy_cuts_${subset}.jsonl.gz
    # fi
fi
