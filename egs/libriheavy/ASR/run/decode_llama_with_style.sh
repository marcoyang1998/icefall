#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="5"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

exp_name=exp_medium_llama-3b_text_encoder_memory_drop_0.05_4gpu_md330_accum_grad_2_with_style_1_with_context_list_0_mask_0.0_pre_text_mixed_style_random
use_ls_test_set=0
use_ls_context_list=0
long_audio_recog=0
use_style_prompt=1

for epoch in 14; do
    for avg in 2; do
        for use_pre_text in 1; do
            for style_trans in "mixed-punc"; do
                for pre_trans in "mixed-punc"; do
                    for ls_distractors in 0; do
                        for max_lens in 1000; do
                            python ./zipformer_prompt_asr/decode_llama_with_style.py \
                                --epoch $epoch \
                                --avg $avg \
                                --use-averaged-model 1\
                                --post-normalization True \
                                --causal 0 \
                                --chunk-size 32 \
                                --left-context-frames 256 \
                                --exp-dir ./zipformer_prompt_asr/$exp_name \
                                --manifest-dir data/fbank \
                                --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
                                --max-duration 750 \
                                --decoding-method greedy_search \
                                --beam-size 4 \
                                --text-encoder-type llama-3b \
                                --long-audio-recog $long_audio_recog \
                                --on-the-fly-feats $long_audio_recog \
                                --use-ls-test-set $use_ls_test_set \
                                --use-ls-context-list $use_ls_context_list \
                                --max-prompt-lens $max_lens \
                                --ls-distractors $ls_distractors \
                                --use-pre-text $use_pre_text \
                                --use-style-prompt $use_style_prompt \
                                --style-text-transform $style_trans \
                                --pre-text-transform $pre_trans \
                                --compute-CER 0
                        done
                    done
                done
            done
        done
    done
done
