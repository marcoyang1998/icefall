#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="2"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

exp_dir=exp_causal_0_medium_subformer_freeze_1_base_lr_0.045_same_bpe_with_context_list_1_more_distractors_2_styles_rerun
use_ls_test_set=0
use_ls_context_list=0
long_audio_recog=0
biasing_level=utterance
causal=0

decoding_method=modified_beam_search


for epoch in 50; do
    for avg in 10; do
        for use_pre_text in 1; do
            for use_style_prompt in 1; do
                for style_trans in "upper-no-punc"; do
                    for pre_trans in "upper-no-punc"; do
                        for ls_distractors in 0; do
                            python ./zipformer_prompt_asr/decode_subformer_with_style.py \
                                --epoch $epoch \
                                --avg $avg \
                                --use-averaged-model True \
                                --post-normalization 1 \
                                --causal $causal \
                                --chunk-size 32 \
                                --left-context-frames 256 \
                                --exp-dir ./zipformer_prompt_asr/${exp_dir} \
                                --manifest-dir data/fbank \
                                --bpe-model /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/data/lang_bpe_500/bpe.model \
                                --text-encoder-bpe-model /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/data/lang_bpe_500/bpe.model \
                                --text-encoder-causal False \
                                --max-duration 1000 \
                                --decoding-method $decoding_method \
                                --load-pretrained False \
                                --use-ls-test-set $use_ls_test_set \
                                --use-ls-context-list $use_ls_context_list \
                                --biasing-level $biasing_level \
                                --ls-distractors $ls_distractors \
                                --long-audio-recog $long_audio_recog \
                                --long-audio-cuts data/long_audio_test_other/long_audio_test_other.jsonl.gz \
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