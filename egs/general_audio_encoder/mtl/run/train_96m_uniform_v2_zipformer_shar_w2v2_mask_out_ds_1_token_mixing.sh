#!/usr/bin/env bash

echo "Num gpus: $GPU_COUNT"

cd /mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl
echo "Current dir: $PWD"

####### Mount the necessary disks #######
bash mount_brainllm_h.sh
ls -lh download/LibriSpeech
#########################################


############## PYTHON env ###############
source /home/housiyuan/miniconda3/etc/profile.d/conda.sh && conda activate encoder

pip install /mnt/shared-storage-user/housiyuan/xiaoyu/software/petrel-oss-sdk-2.3.24.tar.gz

echo "Current Directory: $PWD"

export PETRELOSS_CONFIG=/mnt/shared-storage-user/housiyuan/xiaoyu/petreloss.conf
export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/lhotse_dev_petreloss/:$PYTHONPATH
#########################################


lr=0.04
estimate_epoch=1
lr_hours=30000
lr_batches=7500

# dataset 
use_librispeech=0
full_libri=1
repeat_librispeech=1
use_gigaspeech=1
gigaspeech_subset=xl
use_libriheavy=1
libriheavy_subset=large
use_wenetspeech=0
wenetspeech_subset=L
use_audioset=0
repeat_audioset=1
audioset_subset=full
use_voxpopuli=1
use_yodas=0
use_emotion_dataset=0
repeat_emo=4
use_commonvoice=0
use_mls=0
use_fleurs=0

# augmentation
enable_rir=0
enable_musan=0
min_snr=-5
max_snr=5
mixing_prob=0.5
batch_mixing=1
token_mixing=1
p_noise=0.8
min_noise_snr=5
batch_mix_mode=max
mix_delay_max=2.0

# normalization
enable_spec_aug=0
time_mask_ratio=1.0
mask_mode=w2v2
mask_prob=0.5
mask_length=10
mask_channel_prob=0.25
mask_channel_length=15
loss_only_mask=0

# mvq KD
output_downsampling_factor=1
frame_rate_ratio=1
num_codebooks=16
delta=0
mvq_loss_by_task=0

# at KD
do_audio_tagging=0
audio_tagging_loss_scale=5.0
at_KD=0

# data related
use_shar=1
zip_sampler=0
bucket_sampler=1
num_buckets=20
merge_buckets=1
sync_buckets=0
at_weighted_sampler=0
at_num_samples=400000
max_duration=600

use_fp16=0
use_bf16=1

# lh-${libriheavy_subset}-giga-${gigaspeech_subset}-voxpopuli-${use_voxpopuli}-yodas-${use_yodas}-\
# exp_dir=zipformer_audio_encoder/exp-95m-uni-v2-out-ds-${output_downsampling_factor}-ls-960-lr-${lr}-\
exp_dir=zipformer_audio_encoder/exp-95m-uni-v2-out-ds-${output_downsampling_factor}-lh-giga-vox-\
lr-${lr}-lr-hours-${lr_hours}-${mask_mode}-mask-p-${mask_prob}-l-${mask_length}-\
cha-mask-p-${mask_channel_prob}-l-${mask_channel_length}-\
${batch_mix_mode}-delay-${mix_delay_max}s-token-mix-p-${mixing_prob}-min-snr-${min_snr}-p-n-${p_noise}-min-snr-${min_noise_snr}-\
noise-full-wavlm-large-layer-21-cb-${num_codebooks}-shar-md-${max_duration}-bf16-${use_bf16}

# exp_dir=zipformer_audio_encoder/exp-debug

echo $exp_dir

# --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
# --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \

torchrun --nproc_per_node=$GPU_COUNT --master_port 13409 \
  zipformer_audio_encoder/train_multi_KD3_shar_w2v2_mask_token_mixing.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iter 400000 \
    --keep-last-k 20 \
    --use-shar $use_shar --shar-dir data-shar/data-shar-wavlm-large-layer-21-normalize-cb16-hdf5 \
    --base-lr $lr --estimate-epoch $estimate_epoch --lr-hours $lr_hours --lr-batches $lr_batches \
    --use-fp16 $use_fp16 \
    --use-bf16 $use_bf16 \
    --exp-dir $exp_dir \
    --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --use-librispeech $use_librispeech --full-libri 1 --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-voxpopuli $use_voxpopuli --use-yodas $use_yodas \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --use-mls $use_mls --use-fleurs $use_fleurs --use-commonvoice $use_commonvoice \
    --use-emotion-dataset $use_emotion_dataset --repeat-emo $repeat_emo \
    --do-mvq 1 --mvq-loss-by-task $mvq_loss_by_task \
    --do-audio-tagging $do_audio_tagging --at-KD $at_KD \
    --enable-rir $enable_rir --rir-cuts None \
    --enable-musan $enable_musan \
    --token-mixing $token_mixing \
    --batch-mix-mode $batch_mix_mode --mix-delay-max $mix_delay_max \
    --batch-mixing $batch_mixing --p-noise $p_noise --min-noise-snr $min_noise_snr \
    --mixing-prob $mixing_prob --min-snr $min_snr --max-snr $max_snr \
    --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
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
    --causal 0 \
    --spec-aug-time-warp-factor -1 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --bucketing-sampler $bucket_sampler --num-buckets $num_buckets --merge-buckets $merge_buckets --sync-buckets $sync_buckets \
    --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 12