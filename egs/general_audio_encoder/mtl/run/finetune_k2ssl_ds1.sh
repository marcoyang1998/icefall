#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH

# data related
use_librispeech=1
full_libri=0
use_gigaspeech=0
gigaspeech_subset=s

causal=0
lr=0.02

do_audio_tagging=0
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=zipformer_audio_encoder/exp-full-libri-96M-zipformer-non-streaming-mvq-out-ds-1-mask-ratio-1.0-musan-0-quantizer-v2/iter-224000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=-1
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.1
warmup_batches=0.0
large_batch_count=0

output_ds=1
post_output_ds=2

md=600

exp_dir=zipformer_audio_encoder_finetune/exp-finetune-95M-ls-100h-\
lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-whisper-quantizer-v2-output-ds-1-no-musan-mask-ratio-1.0-224k
# exp_dir=zipformer_audio_encoder_finetune/exp-debug

torchrun --nproc_per_node=8 --master_port=19132 \
  zipformer_audio_encoder/finetune_mtl.py \
    --num-epochs 222 \
    --use-fp16 1 \
    --start-epoch 1 \
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
    --output-downsampling-factor $output_ds \
    --post-encoder-downsampling-factor $post_output_ds \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"