#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

vq_name=vq_hubert_large_layer_21_normalize_1_cb_16

# for subset in dev-clean dev-other; do
#     python local/npy2hdf5.py \
#         --num-jobs 2 \
#         --input-manifest data/${vq_name}/librispeech_cuts_${subset}.jsonl.gz \
#         --manifest-name librispeech_cuts_$subset \
#         --manifest-dir data_hdf5/${vq_name} \
#         --target-manifest-file data_hdf5/${vq_name}/librispeech_cuts_${subset}.jsonl.gz
# done

# for subset in m l; do
#     python local/npy2hdf5.py \
#         --num-jobs 12 \
#         --input-manifest data/${vq_name}/gigaspeech_cuts_${subset}.jsonl.gz \
#         --manifest-name gigaspeech_cuts_$subset \
#         --manifest-dir data_hdf5/${vq_name} \
#         --target-manifest-file data_hdf5/${vq_name}/gigaspeech_cuts_${subset}.jsonl.gz
# done

for subset in xl; do
    num_splits=6
    split_dir=data/$vq_name/giga_${subset}_split
    for i in $(seq 0 1 $(($num_splits-1))); do
        python local/npy2hdf5.py \
            --num-jobs 16 \
            --input-manifest $split_dir/gigaspeech_cuts_${subset}.${i}.processed.jsonl.gz \
            --manifest-name gigaspeech_cuts_${subset}_split_${i} \
            --manifest-dir data_hdf5/${vq_name} \
            --target-manifest-file data_hdf5/${vq_name}/gigaspeech_cuts_${subset}.${i}.processed.jsonl.gz
    done
done

# for subset in large; do
#     python local/npy2hdf5.py \
#         --num-jobs 32 \
#         --input-manifest data/${vq_name}/libriheavy_cuts_${subset}.jsonl.gz \
#         --manifest-name libriheavy_cuts_$subset \
#         --manifest-dir data_hdf5/${vq_name} \
#         --target-manifest-file data_hdf5/${vq_name}/libriheavy_cuts_${subset}.jsonl.gz
# done