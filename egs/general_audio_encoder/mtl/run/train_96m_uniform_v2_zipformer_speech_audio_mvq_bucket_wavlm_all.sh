#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev/:$PYTHONPATH

lr=0.045
lr_hours=25000
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
use_emotion_dataset=0
repeat_emo=4
use_fisher=0
use_mls=0
use_voxpopuli=1
voxpopuli_subset=en_v2

use_audioset=1
repeat_audioset=2
audioset_subset=full
use_music4all=1
use_mtg=1
use_vggsound=1
repeat_vggsound=1
use_bbceffect=1
use_freesound=1
audio_duration_factor=2.0

# augmentation
enable_rir=0
enable_musan=1
batch_mixing=0
min_snr=10
max_snr=20
mixing_prob=0.5
p_noise=0.7
min_noise_snr=5

# normalization
enable_spec_aug=0
time_mask_ratio=1.0
mask_mode=w2v2
mask_prob=0.65
mask_length=10
mask_channel_prob=0.25
mask_channel_length=20
loss_only_mask=0

# audio tagging
do_at=0

# mvq KD
output_downsampling_factor=2
num_codebooks="16,8"
delta="0,0"
frame_rate_ratio="2,1"
num_cb_speech=16
num_cb_audio=8
speech_mvq_loss_scale=1.0
audio_mvq_loss_scale=0.1

# at KD
audio_tagging_loss_scale=5.0
at_KD=1

# data related
use_shar=1
zip_sampler=0
bucket_sampler=1
num_buckets=20
merge_buckets=1
sync_buckets=0
use_custom_duration_bins=0
at_weighted_sampler=0
at_num_samples=400000
max_duration=400


exp_dir=zipformer_audio_encoder/exp-96M-uniform-v2-zipformer-out-ds-${output_downsampling_factor}-\
lh-${libriheavy_subset}-giga-${gigaspeech_subset}-voxpopuli-${use_voxpopuli}-\
as-${audioset_subset}-x${repeat_audioset}-all-audio-\
${mask_mode}-mask-p-${mask_prob}-l-${mask_length}-cha-mask-p-${mask_channel_prob}-l-${mask_channel_length}-\
musan-p-${mixing_prob}-min-snr-${min_snr}-\
multi-mvq-wavlm-all-wavlm-large-cb16-${speech_mvq_loss_scale}-dasheng-cb8-${audio_mvq_loss_scale}-\
md${max_duration}

# batch-mix-p-${mixing_prob}-min-snr-${min_snr}-p-noise-${p_noise}-min-snr-${min_noise_snr}-\

# exp_dir=zipformer_audio_encoder/exp-debug

echo "env info"
echo $MASTER_ADDR
echo $MASTER_PORT
echo $WORLD_SIZE
echo $RANK

echo $exp_dir

torchrun --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  zipformer_audio_encoder/train_multi_KD3_shar_speech_audio_multi_mvq2.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iter 500000 \
    --use-shar $use_shar \
    --speech-shar-dir data-shar/data-shar-wavlm-large-layer-21-normalize-cb16-hdf5 \
    --audio-shar-dir data-shar-cpfs04/data-shar-wavlm-large-l-21-normalize-cb16-dasheng-cb8-combined-hdf5 \
    --base-lr $lr --lr-hours $lr_hours --lr-batches $lr_batches \
    --use-fp16 1 \
    --exp-dir $exp_dir \
    --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --use-librispeech $use_librispeech --full-libri 1 --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --use-voxpopuli $use_voxpopuli --voxpopuli-subset $voxpopuli_subset \
    --use-fisher $use_fisher --use-mls $use_mls \
    --use-emotion-dataset $use_emotion_dataset --repeat-emo $repeat_emo \
    --use-music4all $use_music4all \
    --use-vggsound $use_vggsound --repeat-vggsound $repeat_vggsound \
    --use-freesound $use_freesound --use-bbceffect $use_bbceffect --use-mtg $use_mtg \
    --audio-duration-factor $audio_duration_factor \
    --do-audio-tagging $do_at \
    --at-KD $at_KD --do-mvq 1 \
    --speech-mvq-loss-scale $speech_mvq_loss_scale --audio-mvq-loss-scale $audio_mvq_loss_scale \
    --enable-musan $enable_musan \
    --batch-mixing $batch_mixing --p-noise $p_noise --min-noise-snr $min_noise_snr \
    --mixing-prob $mixing_prob --min-snr $min_snr --max-snr $max_snr \
    --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --spec-aug-time-warp-factor -1 \
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
    --manifest-dir data/vq_whisper_turbo_cb16_firered_zh_en_cb32 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --num-cb-speech $num_cb_speech --num-cb-audio $num_cb_audio \
    --bucketing-sampler $bucket_sampler --num-buckets $num_buckets --merge-buckets $merge_buckets --sync-buckets $sync_buckets \
    --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --on-the-fly-feats 1 \
    --max-duration $max_duration --max-cuts 150 \
    --num-workers 8