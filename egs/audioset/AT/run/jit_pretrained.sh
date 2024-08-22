#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate multi_KD

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="5"

python zipformer/jit_pretrained.py \
    --nn-model-filename zipformer/exp_at_as_full/jit_script.pt \
    --label-dict downloads/audioset/class_labels_indices.csv \
    downloads/audioset/eval/wav/__p-iA312kg_70.000_80.000.wav \
    /star-fj/fangjun/data/librispeech/LibriSpeech/dev-clean/1272/128104/1272-128104-0002.flac