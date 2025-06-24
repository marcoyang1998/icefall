#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=~/xiaoyu/workspace/lhotse:$PYTHONPATH

# data related
use_librispeech=0
full_libri=0
use_audioset=1
repeat_audioset=1
audioset_subset=full

causal=0
lr=0.02

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
frames_mask_size=192
features_mask_size=40


# finetune args
do_finetune=1
finetune_ckpt=zipformer_audio_encoder/exp-96M-zipformer-non-streaming-as-full-dasheng-mvq-cb16-do-at-0-mask-ratio-1.0-musan-1-time-mask-192-feature-mask-40/iter-300000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=-1
encoder_lr_scale=0.1

# sampler
weighted_sampler=1
bucketing_sampler=0
at_num_samples=200000

md=600

exp_dir=zipformer_audio_encoder_finetune/exp-finetune-as-${audioset_subset}\
-lr-${lr}-musan-${enable_musan}-mixup-${mixup_prob}-freeze-encoder-${freeze_encoder}\
-encoder-lr-scale-${encoder_lr_scale}-frames-mask-${frames_mask_size}-features-mask-${features_mask_size}\
-from-dasheng-as-mvq-cb16-larger-mask-300k

# exp_dir=zipformer_audio_encoder_finetune/exp-debug

torchrun --nproc_per_node=8 --master_port=19831 \
  zipformer_audio_encoder/finetune_at.py \
    --num-epochs 90 \
    --start-epoch 1 \
    --use-fp16 1 \
    --save-with-client False \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --exp-dir $exp_dir \
    --manifest-dir data_s3/fbank_as_ced_mAP50 \
    --base-lr $lr \
    --enable-musan $enable_musan \
    --enable-spec-aug $enable_spec_aug --spec-aug-time-warp-factor $time_warp --time-mask-ratio $time_mask_ratio\
    --enable-mixup $enable_mixup --mixup-prob $mixup_prob \
    --frames-mask-size $frames_mask_size --features-mask-size $features_mask_size \
    --do-asr $do_asr \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune $do_finetune --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --bucketing-sampler $bucketing_sampler --at-weighted-sampler $weighted_sampler --at-num-samples $at_num_samples \
    --on-the-fly-feats 1 \
    --num-workers 20 \
    --max-duration $md

echo "Done"