#!/usr/bin/env bash

source ~/softwares/pyenvs/k2_cuda11/k2_cuda11/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="1"

for epoch in $(seq 70 -2 60); do
    for avg in $(seq 40 -2 32); do
        python zipformer/evaluate.py \
	    	--epoch $epoch \
	    	--avg $avg \
			--use-averaged-model 1 \
			--feature-dim 128 \
			--manifest-dir data/fbank_as_ced_mAP50 \
	    	--exp-dir zipformer/exp_at_full_lr_epochs_15_specaug1_frame192_feature27_musan1_weighted1_md1000_bf16 \
	    	--max-duration 500 
    done
done    
