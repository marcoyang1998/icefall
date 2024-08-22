#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate multi_KD

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="0"

python zipformer/pretrained.py \
    --checkpoint zipformer/exp_at_as_full/pretrained.pt \
    --label-dict downloads/audioset/class_labels_indices.csv \
    downloads/audioset/eval/wav/__p-iA312kg_70.000_80.000.wav \
    /star-xy/softwares/icefall_development/icefall_audio_tagging/egs/audioset/AT/downloads/audioset/eval/wav/ZYze8q72FT8_30.000_40.000.wav