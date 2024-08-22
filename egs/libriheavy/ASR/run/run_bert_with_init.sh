#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate prompt_asr

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-data/xiaoyu/icefall_prompt_asr/:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="2,3"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

causal=0
subset="small"
base_lr=0.015
max_duration=1000
memory_dropout_rate=0.05
use_style_prompt=0
text_encoder_type=BERT

memory_layer=0
use_context_list=1
top_k=10000

exp_dir=exp_${subset}_${text_encoder_type}_base_lr_${base_lr}_style_prompt_${use_style_prompt}_mixed_punc_prompts_only_finetune_cross_attnetion_set_eval

backbone_ckpt=/star-data/xiaoyu/icefall_prompt_asr/egs/libriheavy/ASR/zipformer_prompt_asr/exp_baseline_small_causal_0_2gpus_md1000_mixed_trans/pretrained.pt

python ./zipformer_prompt_asr/train_bert_encoder_init.py \
    --world-size 2 \
    --start-epoch 1 \
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
    --backbone-ckpt  $backbone_ckpt \
    --max-duration $max_duration \
    --text-encoder-type $text_encoder_type \
    --use-context-list $use_context_list \
    --top-k $top_k \
    --rare-word-file data/context_biasing/medium_rare_words_topk_${top_k}.txt \
    --use-style-prompt $use_style_prompt \
    --master-port 13992