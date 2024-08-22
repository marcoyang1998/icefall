#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="2"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

exp_name=exp_medium_BERT_text_encoder_memory_drop_0.05_4gpu_md1000_with_style_1_with_context_list_1_pre_text_random_style_random
use_ls_test_set=1
two_pass_decoding=1
ls_distractors=False

for epoch in 40; do
    for avg in 5; do
        for use_pre_text in 1; do
            for style_trans in "upper-no-punc"; do
                for pre_trans in "mixed-punc"; do
                        python ./zipformer_prompt_asr/decode_bert_with_style_2pass.py \
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
                            --max-duration 1000 \
                            --decoding-method greedy_search \
                            --beam-size 4 \
                            --text-encoder-type BERT \
                            --use-ls-test-set $use_ls_test_set \
                            --two-pass-decoding $two_pass_decoding \
                            --ls-distractors $ls_distractors \
                            --use-pre-text $use_pre_text \
                            --use-style-prompt 1 \
                            --style-text-transform $style_trans \
                            --pre-text-transform $pre_trans \
                            --compute-CER 0
                done
            done
        done
    done
done
