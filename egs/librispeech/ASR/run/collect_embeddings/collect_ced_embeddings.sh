#!/usr/bin/env bash

source ~/softwares/pyenvs/k2_cuda11/k2_cuda11/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-data/xiaoyu/icefall_multi_KD:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="4,5"


for part in unbalanced balanced eval; do
    python multi_KD/collect_ced_embeddings.py \
        --num-jobs 2 \
        --max-duration 1000 \
        --input-manifest /star-xy/softwares/icefall_development/icefall_audio_tagging/egs/audioset/AT/data/fbank_as_ced_mAP50/cuts_audioset_${part}.jsonl.gz \
        --output-manifest embeddings-audioset-${part} \
        --ced-ckpt pretrained_models/CED/audiotransformer_base_mAP_4999.pt \
        --model-id CED-base-mAP50 \
        --embedding-dir /star-xy/softwares/icefall_development/icefall_audio_tagging/egs/audioset/AT/data/fbank_as_ced_mAP50
done