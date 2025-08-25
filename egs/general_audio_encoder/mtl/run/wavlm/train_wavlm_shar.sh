#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

optimizer=adamw
lr=5e-4
weight_decay=0.01
warmup_start=0.0
warmup_batches=32000
scheduler=cosine
num_training_steps=300000

# dataset 
use_librispeech=1
full_libri=1
repeat_librispeech=1
use_gigaspeech=0
gigaspeech_subset=m
use_libriheavy=0
libriheavy_subset=large
use_wenetspeech=0
wenetspeech_subset=L
use_audioset=0
repeat_audioset=1
audioset_subset=full

# model related
model_version=base
encoder_dim=768
causal=False
mask_prob=0.65

# augmentation
enable_rir=0
enable_musan=1
enable_spec_aug=0
time_mask_ratio=1.0

# mvq KD
num_codebooks=16
delta=0
frame_rate_ratio=1

# at KD
do_AT=0
audio_tagging_loss_scale=5.0
at_KD=0

# data related
use_shar=1
zip_sampler=0
bucket_sampler=1
at_weighted_sampler=0
at_num_samples=400000
max_duration=300

exp_dir=wavlm_pretrain/exp-wavlm-${model_version}-${optimizer}-lr-${lr}-warm-${warmup_batches}-\
ls-full-mask-prob-${mask_prob}-musan-${enable_musan}-rir-${enable_rir}-\
hubert-large-layer-21-libri-mvq-cb${num_codebooks}-shar

exp_dir=wavlm_pretrain/exp-debug

echo $exp_dir

# torchrun \
#   --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
#   --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
torchrun \
  --nproc_per_node 1 \
  wavlm_pretrain/train_multi_KD3_shar.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iter 300000 \
    --use-shar $use_shar --shar-dir data-shar/data-shar-hubert-large-layer-21-normalize-cb16 \
    --base-lr $lr --opt $optimizer --weight-decay $weight_decay \
    --lr-scheduler $scheduler --num-training-steps $num_training_steps \
    --warmup-batches $warmup_batches --warmup-start $warmup_start \
    --use-fp16 0 \
    --exp-dir $exp_dir \
    --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --use-librispeech $use_librispeech --full-libri 1 --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --at-KD $at_KD --do-mvq 1 --do-audio-tagging 0 \
    --enable-rir $enable_rir --rir-cuts None \
    --enable-musan $enable_musan --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --wavlm-version $model_version \
    --encoder-dim $encoder_dim \
    --mask-prob $mask_prob \
    --causal $causal \
    --manifest-dir data/vq_whisper_turbo_zh_en_16_v2 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --bucketing-sampler $bucket_sampler --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --num-buckets 30 \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 4