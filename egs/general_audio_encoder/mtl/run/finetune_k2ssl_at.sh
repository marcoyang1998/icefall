#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

# data related
use_shar=0
use_librispeech=0
full_libri=0
use_gigaspeech=0
gigaspeech_subset=s
use_audioset=1
repeat_audioset=3
audioset_subset=balanced

causal=0
lr=0.02

do_audio_tagging=1
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=zipformer_audio_encoder/exp-full-libri-96M-zipformer-non-streaming-whisper-dasheng-multi-mvq-cb16-do-at-0-mask-ratio-1.0-musan-1/iter-224000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=-1
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.1
warmup_batches=0.0
large_batch_count=0

md=1200

exp_dir=zipformer_audio_encoder_finetune/exp-finetune-95M-as-${audioset_subset}-\
lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-lh-large-shar-whisper-dasheng-multi-mvq-cb16-musan-mask-ratio-1.0-224k-rerun

# exp_dir=zipformer_audio_encoder_finetune/exp-debug

export CUDA_VISIBLE_DEVICES="2,3"

torchrun --nproc_per_node=2 --master_port=19134 \
  zipformer_audio_encoder/finetune_at.py \
    --num-epochs 50 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-audioset $use_audioset --audioset-subset $audioset_subset \
    --repeat-audioset $repeat_audioset \
    --exp-dir $exp_dir \
    --use-shar $use_shar --shar-dir data-shar/data-shar-whisper-zh-en-cb16-v2 \
    --manifest-dir data/fbank_as_ced_mAP50 \
    --base-lr $lr \
    --do-audio-tagging $do_audio_tagging \
    --at-KD $at_KD \
    --do-finetune 1 --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --warmup-batches $warmup_batches \
    --large-batch-count $large_batch_count \
    --causal $causal \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"