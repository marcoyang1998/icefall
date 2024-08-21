#!/usr/bin/env bash
source /star-xy/.bashrc

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate multi_KD

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_pruning:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="0"
echo "Using device: ${CUDA_VISIBLE_DEVICES}"

for epoch in 30; do
    for avg in 15; do
        python ./zipformer/decode_prune.py \
            --epoch $epoch \
            --avg $avg \
            --dropout-with-probe 1 \
            --exp-dir ./zipformer/exp_probe1 \
            --max-duration 1000 \
            --prune-threshold 0.05 \
            --decoding-method greedy_search
    done
done