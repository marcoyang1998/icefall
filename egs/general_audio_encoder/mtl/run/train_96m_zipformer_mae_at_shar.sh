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
gigaspeech_subset=m
use_libriheavy=0
libriheavy_subset=large
use_wenetspeech=0
wenetspeech_subset=L
use_audioset=1
repeat_audioset=1
audioset_subset=full

# mae related
mae_loss_norm=frame
mae_loss_scale=1.0
mae_downsample_factor=4

# augmentation
enable_rir=0
enable_musan=1
enable_spec_aug=1
time_mask_ratio=1.0
frames_mask_size=192
features_mask_size=40

# mvq KD
output_downsampling_factor=2
num_codebooks=8
delta=0
frame_rate_ratio=1
mvq_loss_scale=0.0

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
max_duration=600

exp_dir=zipformer_audio_encoder/exp-96M-zipformer-mae-as-full-\
out-ds-${output_downsampling_factor}-mask-ratio-${time_mask_ratio}-\
frames-mask-${frames_mask_size}-features-mask-${features_mask_size}-musan-${enable_musan}-rir-${enable_rir}-\
mae-scale-${mae_loss_scale}-norm-${mae_loss_norm}-mvq-${mvq_loss_scale}-\
small-decoder-dasheng-large-as-mvq-shar

# exp_dir=zipformer_audio_encoder/exp-debug

echo $exp_dir
# torchrun \
#   --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
#   --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \

torchrun \
  --nproc_per_node 8 \
  zipformer_audio_encoder/train_multi_KD3_shar_mae.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iter 300000 \
    --start-batch 60000 \
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
    --do-audio-tagging $do_audio_tagging --at-KD $at_KD --do-mvq 1 \
    --enable-rir $enable_rir --rir-cuts None \
    --enable-musan $enable_musan --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --frames-mask-size $frames_mask_size --features-mask-size $features_mask_size \
    --output-downsampling-factor $output_downsampling_factor \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --causal 0 \
    --num-decoder-layers 2,2,2,2 \
    --decoder-dim 192,256,256,256 \
    --decoder-downsampling-factor 1,2,2,2 \
    --decoder-cnn-module-kernel 7,7,7,7 \
    --decoder-feedforward-dim 512,768,768,768 \
    --decoder-unmasked-dim 192,192,192,192 \
    --decoder-num-heads 4,4,4,4 \
    --manifest-dir data/vq_whisper_turbo_zh_en_16_v2 \
    --spec-aug-time-warp-factor -1 \
    --mvq-loss-scale $mvq_loss_scale \
    --mae-loss-norm $mae_loss_norm --mae-loss-scale $mae_loss_scale --mae-downsample-factor $mae_downsample_factor \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --bucketing-sampler $bucket_sampler --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --num-buckets 10 \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 16