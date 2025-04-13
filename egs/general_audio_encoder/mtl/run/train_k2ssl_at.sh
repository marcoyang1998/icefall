#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

# data related
use_librispeech=0
full_libri=0
use_audioset=1
repeat_audioset=3
audioset_subset=balanced

causal=0
lr=0.035

do_finetune=0
do_asr=0
do_audio_tagging=1
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

# data augmentation
enable_musan=0
enable_spec_aug=1
time_warp=80
time_mask_ratio=1.0

freeze_encoder=0
freeze_encoder_steps=-1
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=1.0

md=1000

exp_dir=zipformer_audio_encoder_finetune/exp-AT-as-${audioset_subset}\
-lr-${lr}-causal-${causal}-musan-${enable_musan}-from-multi-mvq-224k

# exp_dir=zipformer_audio_encoder_finetune/exp-debug

export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node=2 --master_port=19130 \
  zipformer_audio_encoder/finetune_mtl.py \
    --num-epochs 60 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank_as_ced_mAP50 \
    --base-lr $lr \
    --enable-musan $enable_musan \
    --enable-spec-aug $enable_spec_aug --spec-aug-time-warp-factor $time_warp --time-mask-ratio $time_mask_ratio\
    --do-asr $do_asr \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune $do_finetune \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"