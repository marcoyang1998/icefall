#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

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
frames_mask_size=100
features_mask_size=27

# finetune args
do_finetune=1
finetune_ckpt=zipformer_audio_encoder/exp-as-full-96M-zipformer-non-streaming-whisper-libri-as-mvq-cb16-dasheng-large-cb16-scale-0.2,1.0-out-ds-2-mask-ratio-1.0-musan-1-frames-mask-192-features-mask-40/iter-300000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=2000
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.05

md=1000

exp_dir=zipformer_audio_encoder_finetune/exp-finetune-as-${audioset_subset}\
-lr-${lr}-causal-${causal}-musan-${enable_musan}-mixup-${mixup_prob}-frames-mask-${frames_mask_size}-features-mask-${features_mask_size}\
-freeze-encoder-${freeze_encoder}-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-AS-full-multi-mvq-whisper-0.2-dasheng-1.0-with-muasn-larger-mask-300k

# exp_dir=zipformer_audio_encoder_finetune/exp-debug

# export CUDA_VISIBLE_DEVICES="2,3"
torchrun --nproc_per_node=2 --master_port=19130 \
  zipformer_audio_encoder/finetune_at.py \
    --num-epochs 30 \
    --start-epoch 1 \
    --use-fp16 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank_as_ced_mAP50 \
    --base-lr $lr \
    --enable-musan $enable_musan \
    --enable-spec-aug $enable_spec_aug --spec-aug-time-warp-factor $time_warp --time-mask-ratio $time_mask_ratio\
    --features-mask-size $features_mask_size --frames-mask-size $frames_mask_size \
    --enable-mixup $enable_mixup --mixup-prob $mixup_prob \
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
    --on-the-fly-feats 1 \
    --num-workers 2 \
    --max-duration $md

echo "Done"