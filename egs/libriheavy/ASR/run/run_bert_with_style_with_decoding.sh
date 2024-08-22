#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="4,5,6,7"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

causal=0
subset="medium"
base_lr=0.045
max_duration=1000
memory_dropout_rate=0.05
use_style_prompt=1
text_encoder_type=BERT

use_context_list=0
bert_adapter=0
context_injection=0
forced_upper_punc=0

exp_dir=exp_${subset}_${text_encoder_type}_text_encoder_memory_drop_${memory_dropout_rate}_4gpu_md${max_duration}_with_style_${use_style_prompt}_with_context_list_${use_context_list}_with_decoding_as_ref_pre_text_random_style_random

python ./zipformer_prompt_asr/train_bert_encoder_with_style_multi_ref.py \
    --world-size 4 \
    --start-epoch 1 \
    --exp-dir ./zipformer_prompt_asr/$exp_dir \
    --use-fp16 True \
    --inf-check False \
    --print-diagnostics False \
    --base-lr $base_lr \
    --memory-dropout-rate $memory_dropout_rate \
    --causal $causal \
    --num-epochs 80 \
    --subset $subset \
    --manifest-dir data/fbank \
    --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
    --max-duration $max_duration \
    --text-encoder-type $text_encoder_type \
    --use-context-list $use_context_list \
    --with-decoding 1 \
    --min-count 10 \
    --rare-word-file data/context_biasing/medium_rare_words_7.txt \
    --use-style-prompt $use_style_prompt \
    --context-injection $context_injection \
    --text-encoder-adapter $bert_adapter \
    --forced-upper-pre-text $forced_upper_punc \
    --master-port 13994