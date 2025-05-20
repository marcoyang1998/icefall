export PYTHONPATH=./../../..:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1"


model_name=whisper

manifest_dir=data/manifests/${model_name}
embedding_dir=data/embeddings/${model_name}
mkdir -p $manifest_dir
mkdir -p $embedding_dir

model_name=turbo
embedding_layer=-1


# for subset in dev-clean dev-other; do
#     python whisper/collect_whisper_embeddings.py \
#         --num-jobs 1 \
#         --input-manifest data/fbank_librispeech/librispeech_cuts_${subset}.jsonl.gz \
#         --manifest-name embeddings-ls-${subset} \
#         --target-manifest-file ${manifest_dir}/whisper-${model_name}-layer-${embedding_layer}-ls-${subset}.jsonl.gz \
#         --embedding-dir $embedding_dir \
#         --embedding-layer $embedding_layer \
#         --max-duration 200 \
#         --whisper-version $model_name
# done

# for subset in balanced eval; do
#     python whisper/collect_whisper_embeddings.py \
#         --num-jobs 2 \
#         --input-manifest data/fbank_as_ced_mAP50/audioset_cuts_${subset}.jsonl.gz \
#         --manifest-name embeddings-as-${subset} \
#         --target-manifest-file ${manifest_dir}/whisper-${model_name}-layer-${embedding_layer}-as-${subset}.jsonl.gz \
#         --embedding-dir $embedding_dir \
#         --embedding-layer $embedding_layer \
#         --max-duration 200 \
#         --whisper-version $model_name
# done

# dataset=giga
# for subset in xs sampled dev; do
#     python whisper/collect_whisper_embeddings.py \
#         --num-jobs 2 \
#         --input-manifest data/fbank_gigaspeech/gigaspeech_cuts_${subset}.jsonl.gz \
#         --manifest-name embeddings-${dataset}-${subset} \
#         --target-manifest-file ${manifest_dir}/whisper-${model_name}-layer-${embedding_layer}-${dataset}-${subset}.jsonl.gz \
#         --embedding-dir $embedding_dir \
#         --embedding-layer $embedding_layer \
#         --max-duration 200 \
#         --whisper-version $model_name
# done

python whisper/collect_whisper_embeddings.py \
    --num-jobs 1 \
    --input-manifest data/fbank_voxceleb/vox1_cuts_test.jsonl.gz \
    --manifest-name embeddings-vox1-test \
    --target-manifest-file ${manifest_dir}/whisper-${model_name}-layer-${embedding_layer}-vox1-test.jsonl.gz \
    --embedding-dir $embedding_dir \
    --embedding-layer $embedding_layer \
    --max-duration 200 \
    --whisper-version $model_name