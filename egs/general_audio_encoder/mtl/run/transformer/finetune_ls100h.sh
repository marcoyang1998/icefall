#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

# data related
use_librispeech=1
full_libri=0
use_gigaspeech=0
gigaspeech_subset=s

# optimizer
lr=5e-4
warmup_batches=1000
opt=adam

# model related
causal=0
num_layers=10
num_heads=8
encoder_dim=768
use_flash_attention=1
causal=0
layerdrop_p=0.1


do_audio_tagging=0
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=transformer/exp-300m-transformer-causal-0-adamw-wd-0.01-lr-1.5e-3-cosine-scheduler-warmup-32000-lh-large-mask-ratio-1.0-musan-1-rir-0-hubert-large-layer-21-libri-mvq-cb16-shar/iter-300000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=2000
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.1

md=1000

exp_dir=transformer_finetune/exp-finetune-95M-ls-100h-${opt}-\
lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-hubert-large-mvq-cb16-with-musan-no-rir-300k

exp_dir=transformer_finetune/exp-debug
echo $exp_dir

# export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node=2 --master_port=19130 \
  transformer/finetune_mtl.py \
    --num-epochs 30 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank \
    --base-lr $lr --opt $opt --warmup-batches $warmup_batches \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune 1 --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --num-layers $num_layers \
    --num-heads $num_heads \
    --encoder-dim $encoder_dim \
    --use-flash-attention $use_flash_attention \
    --layerdrop-prob $layerdrop_p \
    --causal $causal \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"