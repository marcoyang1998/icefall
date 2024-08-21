#!/usr/bin/env bash
source /star-xy/.bashrc

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate multi_KD

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_pruning:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="1"
echo "Using device: ${CUDA_VISIBLE_DEVICES}"

for m in greedy_search modified_beam_search; do
    for epoch in 30; do
        for avg in $(seq 14 -2 4); do
            ./zipformer/decode.py \
                --epoch $epoch \
                --avg $avg \
                --dropout-with-probe 1 \
                --exp-dir ./zipformer/exp_960h_probe1_scheduledfloat \
                --max-duration 1000 \
                --decoding-method $m
        done
    done
done