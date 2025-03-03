#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH

# data related
use_librispeech=1
full_libri=1
use_gigaspeech=0
gigaspeech_subset=s

causal=1
lr=0.02

do_audio_tagging=0
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=zipformer_audio_encoder/exp-large-full-en-lh-large-audio-multi-kd-audio-mvq-scale-0.2-time-mask-ratio-2.0-shar/iter-248000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=-1
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.1
warmup_batches=0.0
large_batch_count=0

md=600

exp_dir=zipformer_audio_encoder_finetune/exp-finetune-ls-960h-\
lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-large-en-mvq-audio-mvq-0.2-lr-0.04-shar-250k


torchrun --nproc_per_node=8 --master_port=19132 \
  zipformer_audio_encoder/finetune_mtl.py \
    --num-epochs 100 \
    --use-fp16 1 \
    --start-epoch 1 \
    --max-iters 100000 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank_mtl \
    --base-lr $lr \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune 1 --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --warmup-batches $warmup_batches \
    --large-batch-count $large_batch_count \
    --causal $causal \
    --chunk-size 8,32,64 \
    --left-context-frames 128,256 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --use-shar 1 --shar-dir data-shar-no-feat \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"