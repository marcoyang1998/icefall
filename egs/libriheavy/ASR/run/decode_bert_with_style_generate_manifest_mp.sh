#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="4,5,6,7"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

exp_name=exp_medium_BERT_text_encoder_memory_drop_0.05_4gpu_md1000_with_style_1_with_context_list_1_pre_text_random_style_random
use_ls_test_set=0

for epoch in 50; do
    for avg in 10; do
        for use_pre_text in 1; do
            for use_style_prompt in 1; do
                for style_trans in "mixed-punc"; do
                    for pre_trans in  "mixed-punc"; do
                            python ./zipformer_prompt_asr/decode_bert_with_style_save_decoding_mp.py \
                                --world-size 4 \
                                --epoch $epoch \
                                --avg $avg \
                                --use-averaged-model True \
                                --post-normalization True \
                                --causal 0 \
                                --chunk-size 32 \
                                --left-context-frames 256 \
                                --exp-dir ./zipformer_prompt_asr/$exp_name \
                                --manifest-dir data/fbank \
                                --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
                                --input-manifest data/fbank/libriheavy_cuts_medium.jsonl.gz \
                                --output-manifest data/manifest_medium_ref_text_as_pre_text_v2 \
                                --log-dir data/manifest_medium_ref_text_as_pre_text_v2 \
                                --max-duration 1400 \
                                --decoding-method greedy_search \
                                --beam-size 4 \
                                --text-encoder-type BERT \
                                --use-ls-test-set $use_ls_test_set \
                                --use-pre-text $use_pre_text \
                                --use-style-prompt $use_style_prompt \
                                --style-text-transform $style_trans \
                                --pre-text-transform $pre_trans \
                                --compute-CER 1
                    done
                done
            done
        done
    done
done
