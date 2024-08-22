#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate prompt_asr

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-data/xiaoyu/icefall_prompt_asr/:$PYTHONPATH
source new_env.sh
export CUDA_VISIBLE_DEVICES="2,3"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

causal=0
subset="small"
base_lr=0.03
max_duration=1000
memory_dropout_rate=0.05
use_style_prompt=0
freeze_text_encoder=1
text_encoder_causal=1

memory_layer=0
use_context_list=0
top_k=10000

exp_dir=exp_${subset}_subformer_base_lr_${base_lr}_style_prompt_${use_style_prompt}_mixed_punc_prompts_only_finetune_cross_attnetion_set_eval

backbone_ckpt=/star-data/xiaoyu/icefall_prompt_asr/egs/libriheavy/ASR/zipformer_prompt_asr/exp_baseline_small_causal_0_2gpus_md1000_mixed_trans/pretrained.pt

python ./zipformer_prompt_asr/train_subformer_init.py \
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
    --bpe-model /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/data/lang_bpe_500/bpe.model \
    --backbone-ckpt  $backbone_ckpt \
    --max-duration $max_duration \
    --freeze-text-encoder $freeze_text_encoder \
    --text-encoder-bpe-model /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/data/lang_bpe_500/bpe.model \
    --text-encoder-ckpt /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/subformer/exp_bert_pretraining_causal_0_with_bpe_2gpu_bs30/pretrained-iter-228000-avg-2.pt \
    --text-encoder-causal $text_encoder_causal \
    --use-context-list $use_context_list \
    --top-k $top_k \
    --rare-word-file data/context_biasing/medium_rare_words_topk_${top_k}.txt \
    --use-style-prompt $use_style_prompt \
    --master-port 13902