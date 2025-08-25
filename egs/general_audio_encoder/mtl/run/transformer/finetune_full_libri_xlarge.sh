#!/usr/bin/env bash

source /root/miniconda3/etc/profile.d/conda.sh && conda activate
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH

# data related
use_librispeech=1
full_libri=1
use_gigaspeech=0
gigaspeech_subset=s

# optimizer
lr=1e-3
optimizer=adamw
warmup_batches=10000
warmup_start=0.0
weight_decay=0.01
scheduler=eden
num_training_steps=200000


# model related
causal=0
num_layers=18
num_heads=16
encoder_dim=1024
use_flash_attention=1


do_audio_tagging=0
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=transformer/exp-300m-transformer-causal-0-adamw-wd-0.01-lr-1.5e-3-cosine-scheduler-warmup-32000-lh-large-mask-ratio-1.0-musan-1-rir-0-hubert-large-layer-21-libri-mvq-cb16-shar/iter-400000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=-1
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.1

md=600

exp_dir=transformer_finetune/exp-finetune-300M-ls-960-${opt}-wd-${weight_decay}-\
sched-${scheduler}-lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-hubert-large-mvq-cb16-with-musan-no-rir-400k

# exp_dir=transformer_finetune/exp-debug
echo $exp_dir

# export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node=8 --master_port=19110 \
  transformer/finetune_mtl.py \
    --num-epochs 100 \
    --max-iters $num_training_steps \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank_librispeech \
    --base-lr $lr --opt $optimizer --weight-decay $weight_decay \
    --lr-scheduler $scheduler --num-training-steps $num_training_steps \
    --warmup-batches $warmup_batches --warmup-start $warmup_start \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune 1 --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --num-layers $num_layers \
    --num-heads $num_heads \
    --encoder-dim $encoder_dim \
    --use-flash-attention $use_flash_attention \
    --causal $causal \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"