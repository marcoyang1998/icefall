#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

# optimizer
optimizer=adamw
lr=5e-4
weight_decay=0.01
warmup_start=0.0
warmup_batches=10000
scheduler=cosine
num_training_steps=50000

freeze_encoder=0
freeze_encoder_steps=5000
encoder_lr_scale=1.0

# data related
use_librispeech=1
full_libri=0
use_gigaspeech=0
gigaspeech_subset=s

do_audio_tagging=0
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

# model related
wavlm_version=base
feature_grad_multi=0.0
encoder_layer_drop=0.05
mask_prob=0.25
mask_channel_prob=0.1

finetune_ckpt=wavlm_pretrain/exp-wavlm-base-adamw-lr-5e-4-warm-32000-ls-full-mask-prob-0.65-musan-1-rir-0-hubert-large-layer-21-libri-mvq-cb16-shar/iter-300000-avg-4.pt

md=250

exp_dir=wavlm_pretrain/exp-finetune-wavlm-${wavlm_version}-ls-100h-opt-${optimizer}\
-lr-${lr}-scheduler-${scheduler}-warmup-${warmup_batches}-layerdrop-${encoder_layer_drop}-freeze-${freeze_encoder_steps}-step\
-from-hubert-large-layer-21-mvq-cb16-mask-prob-0.65-300k-md${md}

# exp_dir=wavlm_pretrain/exp-debug

echo $exp_dir

torchrun --nproc_per_node=8 --master_port=19103 \
  wavlm_pretrain/finetune_asr.py \
    --num-epochs 60 \
    --use-fp16 0 \
    --base-lr $lr --opt $optimizer --weight-decay $weight_decay \
    --lr-scheduler $scheduler --num-training-steps $num_training_steps \
    --warmup-batches $warmup_batches --warmup-start $warmup_start \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank_librispeech \
    --do-audio-tagging $do_audio_tagging \
    --do-finetune 1 --init-modules "encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal 0 \
    --wavlm-version $wavlm_version \
    --feature-grad-mult $feature_grad_multi --encoder-layerdrop $encoder_layer_drop \
    --mask-prob $mask_prob --mask-channel-prob $mask_channel_prob \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"