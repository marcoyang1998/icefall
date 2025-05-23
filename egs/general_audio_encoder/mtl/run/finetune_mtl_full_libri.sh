#!/usr/bin/env bash

export PYTHONPATH=./../../..:$PYTHONPATH

# data related
use_librispeech=1
full_libri=1
use_gigaspeech=0
gigaspeech_subset=s

causal=0
lr=0.02

do_audio_tagging=0
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=zipformer_audio_encoder/exp-96M-zipformer-non-streaming-lh-large-out-ds-2-mask-ratio-1.0-musan-1-rir-0-hubert-large-layer-21-normalized-mvq-cb16-shar/iter-300000-avg-4.pt

enable_musan=1
enable_spec_aug=1
time_warp=80

freeze_encoder=0
freeze_encoder_steps=-1
encoder_lr_scale=0.1
warmup_batches=0.0
large_batch_count=0

md=750

exp_dir=zipformer_audio_encoder_finetune/exp-finetune-ls-960h-\
lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}-time-warp-${time_warp}\
-from-hubert-large-layer-21-normalized-mvq-lh-large-shar-300k

# exp_dir=zipformer_audio_encoder_finetune/exp-debug

torchrun --nproc_per_node=8 --master_port=19132 \
  zipformer_audio_encoder/finetune_mtl.py \
    --num-epochs 60 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank_librispeech \
    --enable-musan $enable_musan --enable-spec-aug $enable_spec_aug \
    --spec-aug-time-warp-factor $time_warp \
    --base-lr $lr \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
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
    --joiner-dim 768 --decoder-dim 768 \
    --use-shar 0 --shar-dir data-shar-no-feat \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"