#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/mnt/petrelfs/zhangchen/xiaoyu/workspace/lhotse:$PYTHONPATH
master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

lr=0.045

# dataset 
use_librispeech=0
full_libri=1
repeat_librispeech=1
use_gigaspeech=0
gigaspeech_subset=xl
use_libriheavy=0
libriheavy_subset=large
use_wenetspeech=0
wenetspeech_subset=L
use_audioset=1
repeat_audioset=1
audioset_subset=full

# augmentation
enable_musan=1
enable_spec_aug=0
mask_mode=w2v2
mask_prob=0.5
mask_length=10
mask_channel_prob=0.25
mask_channel_length=20
loss_only_mask=0

# audio tagging
do_at=0

# mvq KD
output_downsampling_factor=2
num_codebooks=8
delta=0
frame_rate_ratio=1

# at KD
audio_tagging_loss_scale=5.0
at_KD=0

# data related
use_shar=1
zip_sampler=0
bucket_sampler=1
at_weighted_sampler=0
at_num_samples=400000
max_duration=600

exp_dir=zipformer_audio_encoder/exp-96M-zipformer-non-streaming-as-${audioset_subset}\
-${mask_mode}-mask-prob-${mask_prob}-length-${mask_length}\
-channel-mask-prob-${mask_channel_prob}-len-${mask_channel_length}-loss-only-mask-${loss_only_mask}-musan-${enable_musan}\
-dasheng-large-layer--1-as-mvq-cb8

# exp_dir=zipformer_audio_encoder/exp-debug

echo $exp_dir

# torchrun \
#   --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
#   --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
torchrun --nproc_per_node=8 --master_port=19138 --node_rank $SLURM_PROCID --master_addr $master_addr\
  zipformer_audio_encoder/train_multi_KD3_shar_w2v2_mask.py \
    --num-epochs 1 \
    --start-epoch 1 \
    --max-iter 300000 \
    --use-shar $use_shar --shar-dir data-shar/data-shar-dasheng-large-layer--1-normalize-0-as-cb8 \
    --base-lr $lr \
    --use-fp16 1 \
    --exp-dir $exp_dir \
    --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --use-librispeech $use_librispeech --full-libri 1 --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --do-audio-tagging $do_at \
    --at-KD $at_KD --do-mvq 1 \
    --enable-musan $enable_musan --enable-spec-aug $enable_spec_aug \
    --loss-only-mask $loss_only_mask \
    --mask-mode $mask_mode --mask-prob $mask_prob --mask-length $mask_length \
    --mask-channel-prob $mask_channel_prob --mask-channel-length $mask_channel_length \
    --output-downsampling-factor $output_downsampling_factor \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --causal 0 \
    --manifest-dir data/vq_whisper_turbo_cb16_firered_zh_en_cb32 \
    --spec-aug-time-warp-factor -1 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --bucketing-sampler $bucket_sampler --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --num-buckets 10 \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --mvq-loss-by-task False \
    --num-workers 20