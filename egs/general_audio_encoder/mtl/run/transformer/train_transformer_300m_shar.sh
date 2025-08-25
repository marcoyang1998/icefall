#!/usr/bin/env bash

source /root/miniconda3/etc/profile.d/conda.sh && conda activate
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH

# optimizer & scheduler
lr=1.5e-3
optimizer=adamw
warmup_batches=32000
warmup_start=0.0
weight_decay=0.01
scheduler=cosine
num_training_steps=400000

# dataset 
use_librispeech=0
full_libri=1
repeat_librispeech=1
use_gigaspeech=0
gigaspeech_subset=xl
use_libriheavy=1
libriheavy_subset=large
use_wenetspeech=0
wenetspeech_subset=L
use_audioset=0
repeat_audioset=1
audioset_subset=full

# model related
causal=0
num_layers=18
num_heads=16
encoder_dim=1024
use_flash_attention=1

# augmentation
enable_rir=0
enable_musan=1
enable_spec_aug=1
time_mask_ratio=1.0

# mvq KD
num_codebooks=16
delta=0
frame_rate_ratio=1

# at KD
do_audio_tagging=0
audio_tagging_loss_scale=5.0
at_KD=0

# data related
use_shar=1
zip_sampler=0
bucket_sampler=1
at_weighted_sampler=0
at_num_samples=400000
max_duration=400

exp_dir=transformer/exp-300m-transformer-causal-${causal}-${optimizer}-wd-${weight_decay}-\
lr-${lr}-${scheduler}-scheduler-warmup-${warmup_batches}-\
lh-${libriheavy_subset}-\
mask-ratio-${time_mask_ratio}-musan-${enable_musan}-rir-${enable_rir}-\
hubert-large-layer-21-libri-mvq-cb${num_codebooks}-shar

# exp_dir=transformer/exp-debug

echo "env info"
echo $MASTER_ADDR
echo $MASTER_PORT
echo $WORLD_SIZE
echo $RANK

echo $exp_dir

# torchrun \
#   --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
#   --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
torchrun \
    --nproc_per_node=8 --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  transformer/train_multi_KD3_shar.py \
    --num-epochs 20 \
    --save-with-client False \
    --start-epoch 1 \
    --max-iter 400000 \
    --use-shar $use_shar --shar-dir data-shar/data-shar-hubert-large-layer-21-normalize-cb16 \
    --base-lr $lr --opt $optimizer --weight-decay $weight_decay \
    --lr-scheduler $scheduler --num-training-steps $num_training_steps \
    --warmup-batches $warmup_batches --warmup-start $warmup_start \
    --use-fp16 1 \
    --exp-dir $exp_dir \
    --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --use-librispeech $use_librispeech --full-libri 1 --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --do-audio-tagging $do_audio_tagging --at-KD $at_KD --do-mvq 1 \
    --enable-rir $enable_rir --rir-cuts None \
    --enable-musan $enable_musan --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --num-layers $num_layers \
    --num-heads $num_heads \
    --encoder-dim $encoder_dim \
    --use-flash-attention $use_flash_attention \
    --causal $causal \
    --manifest-dir data/vq_hubert_large_layer_21_normalize_1_cb_16 \
    --spec-aug-time-warp-factor -1 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --bucketing-sampler $bucket_sampler --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --num-buckets 30 \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 6