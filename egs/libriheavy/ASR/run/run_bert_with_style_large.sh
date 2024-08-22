#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate prompt_asr

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-data/xiaoyu/icefall_prompt_asr/:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

causal=0
subset="large"
base_lr=0.045
max_duration=1000
memory_dropout_rate=0.05
use_style_prompt=1
text_encoder_type=BERT

memory_layer=0
use_context_list=0
top_k=15000

exp_dir=exp_${subset}_BERT_memory_layer_${memory_layer}_memory_drop_0.05_md1000_with_style_1_with_context_list_${use_context_list}_2_styles_fixed_upper_fixed_BERT_rerun_8GPUs
# exp_dir=exp_medium_BERT_memory_layer_0_memory_drop_0.05_md1000_with_style_1_with_context_list_1_2_styles_fixed_upper_fixed_BERT

python ./zipformer_prompt_asr/train_bert_encoder.py \
    --world-size 8 \
    --start-epoch 8 \
    --exp-dir ./zipformer_prompt_asr/$exp_dir \
    --use-fp16 True \
    --inf-check False \
    --print-diagnostics False \
    --base-lr $base_lr \
    --memory-layer $memory_layer \
    --memory-dropout-rate $memory_dropout_rate \
    --causal $causal \
    --num-epochs 30 \
    --subset $subset \
    --manifest-dir data/fbank \
    --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
    --max-duration $max_duration \
    --text-encoder-type $text_encoder_type \
    --use-context-list $use_context_list \
    --top-k $top_k \
    --rare-word-file data/context_biasing/large_rare_words_topk_15000.txt \
    --use-style-prompt $use_style_prompt \
    --master-port 13990
