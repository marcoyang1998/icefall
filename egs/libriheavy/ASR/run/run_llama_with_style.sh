#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

causal=0
subset="medium"
base_lr=0.045
max_duration=330
accum_grad=2
memory_dropout_rate=0.05
use_style_prompt=1
text_encoder_type="llama-3b"

use_context_list=0
bert_adapter=0
context_injection=0
forced_upper_punc=0
min_count=10

exp_dir=exp_${subset}_${text_encoder_type}_text_encoder_memory_drop_${memory_dropout_rate}_4gpu_md${max_duration}_accum_grad_${accum_grad}_with_style_${use_style_prompt}_with_context_list_${use_context_list}_mask_0.0_pre_text_mixed_style_random

python ./zipformer_prompt_asr/train_llama_encoder_with_style2.py \
    --world-size 6 \
    --start-epoch 1 \
    --exp-dir ./zipformer_prompt_asr/$exp_dir \
    --use-fp16 True \
    --inf-check False \
    --print-diagnostics False \
    --base-lr $base_lr \
    --memory-dropout-rate $memory_dropout_rate \
    --causal $causal \
    --num-epochs 60 \
    --accum-grad $accum_grad \
    --subset $subset \
    --manifest-dir data/fbank \
    --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
    --max-duration $max_duration \
    --text-encoder-type $text_encoder_type \
    --use-context-list $use_context_list \
    --min-count $min_count \
    --rare-word-file data/context_biasing/medium_rare_words_${min_count}.txt \
    --use-style-prompt $use_style_prompt \
    --context-injection $context_injection \
    --text-encoder-adapter $bert_adapter \
    --forced-upper-pre-text $forced_upper_punc \
    --master-port 13992