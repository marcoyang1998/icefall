#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="2"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

exp_name=exp_medium_BERT_text_encoder_baselr_0.045_memory_drop_0.05_4gpu_md1000_with_style_1_joiner_context_injection_0_bert_adapter_0_pre_text_random_style_random

    
for use_pre_text in 1; do
    for use_gt_pre_text in 0; do
        for num_history in 5; do
            python ./zipformer_prompt_asr/transcribe.py \
                --exp-dir zipformer_prompt_asr/${exp_name} \
                --epoch 50 \
                --avg 5 \
                --num-history $num_history \
                --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
                --manifest-dir data/manifest_npr/npr1_cuts_all_guids_0.jsonl.gz \
                --text-encoder-type BERT \
                --use-style-prompt 1 \
                --pre-text-transform mixed-punc \
                --style-text-transform mixed-punc \
                --use-pre-text $use_pre_text \
                --use-gt-pre-text $use_gt_pre_text \
                --causal 0
        done
	done
done
