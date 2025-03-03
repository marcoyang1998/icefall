export PYTHONPATH=./../../..:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"


manifest_dir=data/manifests
embedding_dir=data/embeddings

model_name=turbo
embedding_layer=-1


python whisper/collect_whisper_embeddings.py \
    --num-jobs 1 \
    --input-manifest weread-00-subset_cuts.jsonl.gz \
    --manifest-name embeddings-weread-00-subset \
    --target-manifest-file ${manifest_dir}/whisper-${model_name}-layer-${embedding_layer}-weread-00-subset.jsonl.gz \
    --embedding-dir $embedding_dir \
    --embedding-layer $embedding_layer \
    --max-duration 200 \
    --whisper-version $model_name