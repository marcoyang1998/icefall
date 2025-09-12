#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev/:$PYTHONPATH

lr=0.045
lr_hours=20000

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
use_music4all=0
repeat_music4all=1
use_vggsound=0
repeat_vggsound=1
use_bbceffect=0
use_freesound=0
use_mtg=0

# augmentation
enable_rir=0
enable_musan=1
enable_spec_aug=0
time_mask_ratio=1.0
mask_mode=w2v2
mask_prob=0.65
mask_length=10
mask_channel_prob=0.25
mask_channel_length=20
loss_only_mask=0


# mvq KD
output_downsampling_factor=2
num_codebooks=8
delta=0
target_frame_rate=25
frame_rate_ratio=1

# at KD
do_at=0
audio_tagging_loss_scale=5.0
at_KD=0

# data related
use_shar=1
zip_sampler=0
bucket_sampler=1
num_buckets=20
sync_buckets=0
merge_buckets=1
at_weighted_sampler=0
at_num_samples=400000
max_duration=600

exp_dir=zipformer_audio_encoder/exp-96M-uniform-v2-zipformer-lr-hours-${lr_hours}-\
as-${audioset_subset}-\
${mask_mode}-mask-prob-${mask_prob}-mask-len-${mask_length}-\
channel-mask-prob-${mask_channel_prob}-len-${mask_channel_length}-\
atst-mvq-cb8-musan-mix-shar

# exp_dir=zipformer_audio_encoder/exp-debug

echo $exp_dir

# torchrun \
#   --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
#   --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
torchrun --nproc_per_node=8 --master_port=19132 \
  zipformer_audio_encoder/train_multi_KD3_shar_w2v2_mask.py \
    --num-epochs 1 \
    --start-epoch 1 \
    --max-iter 300000 \
    --use-shar $use_shar --shar-dir data-shar/data-shar-atst-normalize-0-as-cb8 \
    --manifest-dir data/vq_atst_frame_layer_-1_normalize_0_cb_8 \
    --base-lr $lr --lr-hours $lr_hours \
    --use-fp16 1 \
    --exp-dir $exp_dir \
    --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --use-librispeech $use_librispeech --full-libri 1 --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --use-music4all $use_music4all --repeat-music4all $repeat_music4all \
    --use-vggsound $use_vggsound --repeat-vggsound $repeat_vggsound \
    --use-freesound $use_freesound \
    --use-bbceffect $use_bbceffect \
    --use-mtg $use_mtg \
    --do-audio-tagging $do_at --at-KD $at_KD \
    --do-mvq 1 --mvq-loss-by-task False \
    --enable-musan $enable_musan --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --loss-only-mask $loss_only_mask \
    --mask-mode $mask_mode --mask-prob $mask_prob --mask-length $mask_length \
    --mask-channel-prob $mask_channel_prob --mask-channel-length $mask_channel_length \
    --output-downsampling-factor $output_downsampling_factor \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,3,3,1,1,1 \
    --feedforward-dim 1536,1536,1536,1536,1536,1536,1536 \
    --encoder-dim 512,512,512,512,512,512,512 \
    --encoder-unmasked-dim 256,256,256,256,256,256,256 \
    --num-heads 8,8,8,8,8,8,8 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-codebooks $num_codebooks \
    --causal 0 \
    --num-codebooks $num_codebooks --distillation-delta $delta \
    --teacher-frame-ratio $frame_rate_ratio --target-frame-rate $target_frame_rate \
    --bucketing-sampler $bucket_sampler --num-buckets $num_buckets --sync-buckets $sync_buckets --merge-buckets $merge_buckets \
    --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 8