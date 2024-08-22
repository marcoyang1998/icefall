#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="3,4,5,6"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

causal=1
text_encoder_causal=0
subset="medium"
base_lr=0.045
max_duration=1000
memory_dropout_rate=0.05
freeze_text_encoder=1

with_context_list=0
min_count=460

exp_dir=exp_causal_${causal}_${subset}_subformer_freeze_${freeze_text_encoder}_base_lr_${base_lr}_same_bpe_with_context_list_${with_context_list}_more_distractors_2_styles_rerun

python ./zipformer_prompt_asr/train_subformer_with_style.py \
    --world-size 4 \
    --start-epoch 37 \
    --exp-dir ./zipformer_prompt_asr/${exp_dir} \
    --use-fp16 True \
    --inf-check False \
    --print-diagnostics False \
    --base-lr $base_lr \
    --memory-dropout-rate $memory_dropout_rate \
    --causal $causal \
    --num-epochs 50 \
    --subset $subset \
    --use-context-list $with_context_list \
    --min-count $min_count \
    --rare-word-file data/context_biasing/medium_rare_words_${min_count}.txt \
    --bpe-model /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/data/lang_bpe_500/bpe.model \
    --load-pretrained True \
    --freeze-text-encoder $freeze_text_encoder \
    --text-encoder-bpe-model /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/data/lang_bpe_500/bpe.model \
    --text-encoder-ckpt /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/subformer/exp_bert_pretraining_causal_0_with_bpe_2gpu_bs30/pretrained-iter-228000-avg-2.pt \
    --text-encoder-causal $text_encoder_causal \
    --manifest-dir data/fbank \
    --max-duration $max_duration \
    --use-style-prompt True \
    --prompt-mask-prob 0.0 \
    --pre-text-shuffle-prob 0.0 \
    --master-port 13993

# --exp-dir ./pruned_transducer_stateless7_libri_heavy/exp_${subset}_subformer_text_encoder_baselr_${base_lr}_lr_scale_0.75_memory_drop_${memory_dropout_rate}_2gpu_md${max_duration}_smaller_with_style \