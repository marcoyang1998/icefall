#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH

lr=0.045

# dataset 
use_librispeech=1
full_libri=1
repeat_librispeech=1
use_gigaspeech=0
gigaspeech_subset=xl
use_libriheavy=0
libriheavy_subset=medium
use_wenetspeech=0
wenetspeech_subset=L
use_audioset=0
repeat_audioset=2
audioset_subset=full

# augmentation
enable_musan=0
enable_spec_aug=1
time_mask_ratio=1.0

# mvq KD
output_downsampling_factor=1
num_codebooks=16
delta=0
frame_rate_ratio=1

# at KD
audio_tagging_loss_scale=5.0
at_KD=1

# data related
use_shar=1
zip_sampler=0
bucket_sampler=1
at_weighted_sampler=0
at_num_samples=400000
max_duration=600

exp_dir=zipformer_audio_encoder/exp-full-libri-96M-zipformer-non-streaming-\
mvq-out-ds-${output_downsampling_factor}-mask-ratio-${time_mask_ratio}-musan-${enable_musan}-quantizer-v2

echo $exp_dir

torchrun \
  --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
  --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
  zipformer_audio_encoder/train_multi_KD3_shar.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iter 300000 \
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
    --at-KD 1 --do-mvq 1 \
    --enable-musan $enable_musan --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --output-downsampling-factor $output_downsampling_factor \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --causal 0 \
    --manifest-dir data/vq_whisper_turbo_zh_en_16_v2 \
    --spec-aug-time-warp-factor -1 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --bucketing-sampler $bucket_sampler --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --num-buckets 30 \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 2