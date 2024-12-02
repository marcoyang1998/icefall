export PYTHONPATH=/mnt/workspace/xiaoyu/workspace/icefall_multi_kd:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"


manifest_dir=data/embeddings

model_name=turbo

for key in test-clean; do
    for embedding_layer in -1; do
        python multi_KD/collect_whisper_embeddings.py \
            --num-jobs 1 \
            --input-manifest data/manifests/librispeech_cuts_${key}.jsonl.gz \
            --manifest-name embeddings-${key} \
            --target-manifest-file ${manifest_dir}/librispeech_cuts_${key}.jsonl.gz \
            --embedding-layer $embedding_layer \
            --max-duration 500 \
            --whisper-version $model_name
    done
done