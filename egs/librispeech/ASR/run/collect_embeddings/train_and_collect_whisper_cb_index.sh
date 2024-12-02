export PYTHONPATH=/mnt/workspace/xiaoyu/workspace/icefall_multi_kd:$PYTHONPATH

export CUDA_VISIBLE_DEVICES="5"
num_codebooks=4

# 1. Train the quantizer
python local/train_mvq.py \
    --embedding-dim 1280 \
    --num-codebooks $num_codebooks \
    --feature-type h5 \
    --embedding-path data/embeddings/whisper-turbo-layer--1-embeddings-mix.h5 \
    --quantizer-path data/quantizer/whisper-turbo-cb-${num_codebooks}.pt

# 2. Extract the codebook indexes on all the subsets
for subset in dev-clean dev-other test-clean test-other train-clean-100; do
    python multi_KD/extract_whisper_mvq.py \
        --num-jobs 1 \
        --input-manifest data/manifests/librispeech_cuts_${subset}.jsonl.gz \
        --target-manifest-file data/vq_whisper_turbo_${num_codebooks}/librispeech_cuts_${subset}.jsonl.gz \
        --n-mels 128 \
        --embedding-dim 1280 \
        --num-codebooks $num_codebooks \
        --manifest-name codebook-indexes-${subset} \
        --embedding-dir data/vq_whisper_turbo_${num_codebooks} \
        --embedding-layer -1 \
        --quantizer-path data/quantizer/whisper-turbo-cb-${num_codebooks}.pt \
        --max-duration 300
done