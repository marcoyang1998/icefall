#!/usr/bin/env bash

source ~/softwares/pyenvs/k2_cuda11/k2_cuda11/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_tagging:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="2"

# for epoch in $(seq 150 -2 148); do
#     for avg in $(seq 26 -2 16); do
#         python zipformer_co_training/evaluate.py \
# 	    	--epoch $epoch \
# 	    	--avg $avg \
# 			--output-downsampling-factor 2 \
# 			--feature-dim 128 \
# 			--manifest-dir data/fbank_as_ced_fixed_4jobs_128D \
# 	    	--exp-dir zipformer_co_training/exp_KD_at_as_full_co_training_extra_proj_segment_level_segment_length_10_scale0.7_segment_scale0.1_aux_scale0.1_matched_specaug_num_masks1_mask_span192_feature_mask27_lr_epochs30_weighted1_md1000-128D \
# 	    	--max-duration 500 
#     done
# done    

for epoch in $(seq 128 -2 120); do
    for avg in $(seq 22 -2 12); do
        python zipformer_co_training/evaluate.py \
	    	--epoch $epoch \
	    	--avg $avg \
			--output-downsampling-factor 2 \
			--use-beats 0 \
			--feature-dim 128 \
			--manifest-dir data/fbank_as_ced_fixed_4jobs_128D \
	    	--exp-dir zipformer_co_training/exp_KD_CED_base_at_as_full_co_training_extra_proj_segment_level_segment_length_10_scale0.7_segment_scale0.1_aux_scale0.1_matched_specaug_num_masks1_mask_span192_feature_mask27_lr_epochs30_weighted1_md1000-128D \
	    	--max-duration 500 
    done
done    


# for epoch in 130; do
#     for avg in 30; do
#         python zipformer_co_training/evaluate.py \
# 	    	--epoch $epoch \
# 	    	--avg $avg \
# 			--output-downsampling-factor 2 \
# 			--use-beats 1 \
# 	    	--exp-dir zipformer_co_training/exp_KD_beats_at_as_full_co_training_extra_proj_segment_level_segment_length_10_scale0.7_segment_scale0.1_aux_scale0.1_matched_specaug_num_masks1_mask_span192_feature_mask27_lr_epochs30_weighted1_md1000 \
# 	    	--max-duration 500 
#     done
# done    
