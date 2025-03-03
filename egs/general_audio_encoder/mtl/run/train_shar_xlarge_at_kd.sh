#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH

causal=1
lr=0.025

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
use_mls=0
use_extra_chinese_dataset=0
use_extra_english_dataset=0

# augmentation
enable_spec_aug=1
time_mask_ratio=1.0

# mvq KD
num_codebooks=-1
delta=3
frame_rate_ratio=2

# at KD
audio_tagging_loss_scale=1.0
at_KD=1

# data related
use_shar=1
zip_sampler=0
bucket_sampler=1
at_weighted_sampler=0
at_num_samples=400000
max_duration=300

# exp_dir=zipformer_audio_encoder/exp-lr-${lr}-causal-ls-giga-${gigaspeech_subset}-lh-${libriheavy_subset}\
# -as-${audioset_subset}-mvq-kd-cb-${num_codebooks}\
# -at-kd-scale-${audio_tagging_loss_scale}-md-${max_duration}-fp16-single-batch

exp_dir=zipformer_audio_encoder/exp-xlarge-lr-${lr}-at-kd-time-mask-ratio-${time_mask_ratio}-shar-md-${max_duration}
# exp_dir=zipformer_audio_encoder/exp-debug

echo $exp_dir

torchrun \
  --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
  --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
  zipformer_audio_encoder/train_at_KD.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iters 200000 \
    --use-shar $use_shar --shar-dir data-shar-no-feat \
    --base-lr $lr \
    --use-fp16 1 \
    --exp-dir $exp_dir \
    --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --use-librispeech $use_librispeech --full-libri 1 --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --use-mls $use_mls \
    --use-extra-chinese-dataset $use_extra_chinese_dataset --use-extra-english-dataset $use_extra_english_dataset \
    --at-KD 1 --do-mvq 0 \
    --enable-musan 0 --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --causal $causal --chunk-size 8,32,64,-1 --left-context-frames 128,256,-1 \
    --manifest-dir data/fbank_mtl \
    --spec-aug-time-warp-factor -1 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --bucketing-sampler $bucket_sampler --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --num-buckets 30 \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 2