#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate prompt_asr

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-data/xiaoyu/icefall_prompt_asr/:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="4,5,6,7"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

causal=0
subset="medium"
base_lr=0.045
max_duration=1000
memory_dropout_rate=0.05
use_style_prompt=1
text_encoder_type=BERT

memory_layer=0
use_context_list=0
top_k=10000

exp_dir=exp_${subset}_${text_encoder_type}_memory_layer_${memory_layer}_memory_drop_${memory_dropout_rate}_md${max_duration}_with_style_${use_style_prompt}_with_context_list_${use_context_list}_2_styles_fixed_upper_fixed_BERT

python ./zipformer_prompt_asr/train_bert_encoder.py \
    --world-size 1 \
    --start-epoch 1 \
    --exp-dir ./zipformer_prompt_asr/$exp_dir \
    --use-fp16 True \
    --inf-check False \
    --print-diagnostics False \
    --base-lr $base_lr \
    --memory-layer $memory_layer \
    --memory-dropout-rate $memory_dropout_rate \
    --causal $causal \
    --num-epochs 60 \
    --subset $subset \
    --manifest-dir data/fbank \
    --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
    --max-duration $max_duration \
    --text-encoder-type $text_encoder_type \
    --use-context-list $use_context_list \
    --top-k $top_k \
    --rare-word-file data/context_biasing/medium_rare_words_topk_${top_k}.txt \
    --use-style-prompt $use_style_prompt \
    --master-port 13994