#!/usr/bin/env bash

source ~/anaconda3/bin/activate && conda activate encoder
source /mnt/cache/share_data/housiyuan/softwares/activate-cuda-11.8.sh

# export PYTHONPATH=/mnt/cache/share_data/housiyuan/icefall_audio_encoder:$PYTHONPATH
export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH

master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

causal=1
lr=0.04

# dataset 
use_librispeech=1
full_libri=1
repeat_librispeech=1
use_gigaspeech=1
gigaspeech_subset=xl
use_libriheavy=1
libriheavy_subset=large
use_wenetspeech=1
wenetspeech_subset=L
use_audioset=1
repeat_audioset=6
audioset_subset=full
use_mls=1
use_extra_chinese_dataset=1
use_extra_english_dataset=1
use_weread=1

# augmentation
enable_musan=1
enable_rir=1
enable_spec_aug=1
time_mask_ratio=1.0

# mvq KD
output_downsampling_factor=2
num_codebooks="16,16"
delta="3,3"
frame_rate_ratio="2,1"
mvq_loss_scales="1.0,1.0"

# at KD
audio_tagging_loss_scale=5.0
at_KD=1

# data related
use_shar=1
zip_sampler=1
bucket_sampler=0
simple_sampler=0
at_weighted_sampler=0
at_num_samples=400000
max_duration=320

exp_dir=zipformer_audio_encoder/exp-lr-${lr}-causal-ls-giga-${gigaspeech_subset}-lh-${libriheavy_subset}\
-mls-${use_mls}-extra-zh-en-use-weread-${use_weread}-as-${audioset_subset}-multi-mvq-kd\
-at-kd-scale-${audio_tagging_loss_scale}-whisper-all-firered-zh\
-bucket-sampler-md-${max_duration}-fix

# exp_dir=zipformer_audio_encoder/exp-xlarge-lr-${lr}-full-en-zh-baoxiang-data-audio-\
# multi-kd-time-mask-ratio-${time_mask_ratio}-shar
# exp_dir=zipformer_audio_encoder/exp-debug-multi-mvq-${enable_musan}-rir-${enable_rir}

# exp_dir=zipformer_audio_encoder/exp-debug

echo $exp_dir

# torchrun \
#   --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
#   --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
echo $SLURM_PROCID
echo $master_addr

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"

torchrun \
  --nproc_per_node 8 --nnodes 4 --node_rank $SLURM_PROCID --master_addr $master_addr\
  zipformer_audio_encoder/train_multi_KD3_shar_multi_mvq3.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iters 500000 \
    --use-shar $use_shar \
    --en-shar-dir data-shar/data-shar-whisper-zh-en-cb16-v2 \
    --zh-shar-dir data-shar/data-shar-firered-en-zh-cb16-v2 \
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
    --use-weread $use_weread \
    --at-KD 1 --do-mvq 1 \
    --enable-musan $enable_musan --enable-rir $enable_rir \
    --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --causal $causal --chunk-size 8,32,64,-1 --left-context-frames 128,256,-1 \
    --manifest-dir data_s3/vq_whisper_turbo_zh_en_16_v2 \
    --spec-aug-time-warp-factor -1 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --mvq-loss-scales $mvq_loss_scales \
    --bucketing-sampler $bucket_sampler --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --num-buckets 8 \
    --simple-sampler $simple_sampler \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 20