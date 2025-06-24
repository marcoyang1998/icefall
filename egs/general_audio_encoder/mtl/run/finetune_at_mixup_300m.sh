#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=~/xiaoyu/workspace/lhotse:$PYTHONPATH

# data related
use_librispeech=0
full_libri=0
use_audioset=1
repeat_audioset=3
audioset_subset=balanced

causal=0
lr=0.045

do_asr=0
do_audio_tagging=1
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

# data augmentation
enable_musan=0
enable_mixup=1
mixup_prob=0.5
enable_spec_aug=1
time_warp=80
time_mask_ratio=1.0


# finetune args
do_finetune=1
finetune_ckpt=zipformer_audio_encoder/exp-300M-zipformer-non-streaming-as-full-dasheng-large-layer--1-as-mvq-cb8-do-at-0-mask-ratio-1.0-musan-1-time-mask-192-feature-mask-40/iter-224000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=2000
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.05

md=1000

exp_dir=zipformer_audio_encoder_finetune/exp-300m-finetune-AT-as-${audioset_subset}\
-lr-${lr}-musan-${enable_musan}-mixup-${mixup_prob}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-dasheng-as-mvq-cb8-with-musan-larger-mask-224k

# exp_dir=zipformer_audio_encoder_finetune/exp-debug

# export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node=2 --master_port=19133 \
  zipformer_audio_encoder/finetune_at.py \
    --num-epochs 30 \
    --start-epoch 1 \
    --use-fp16 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --exp-dir $exp_dir \
    --manifest-dir data_s3/vq_dasheng_large_cb_16 \
    --base-lr $lr \
    --enable-musan $enable_musan \
    --enable-spec-aug $enable_spec_aug --spec-aug-time-warp-factor $time_warp --time-mask-ratio $time_mask_ratio\
    --enable-mixup $enable_mixup --mixup-prob $mixup_prob \
    --do-asr $do_asr \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune $do_finetune --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --on-the-fly-feats 1 \
    --num-workers 16 \
    --max-duration $md

echo "Done"