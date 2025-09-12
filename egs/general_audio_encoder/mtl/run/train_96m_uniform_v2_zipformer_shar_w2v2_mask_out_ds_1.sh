#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH

lr=0.045
estimate_epoch=1
lr_hours=10000
lr_batches=7500

# dataset 
use_librispeech=1
full_libri=1
repeat_librispeech=1
use_gigaspeech=0
gigaspeech_subset=xl
use_libriheavy=0
libriheavy_subset=large
use_wenetspeech=0
wenetspeech_subset=L
use_audioset=0
repeat_audioset=1
audioset_subset=full
use_voxpopuli=0
use_emotion_dataset=0
repeat_emo=4

# musan mixing 
enable_rir=0
enable_musan=0
min_snr=-5
max_snr=20
mixing_prob=0.2

# batch mixing
batch_mixing=0
p_noise=0.5
min_noise_snr=5
mixing_mode=batch

# augmentation
normalize_fbank=0
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
num_codebooks=32
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
max_duration=800

# lh-${libriheavy_subset}-giga-${gigaspeech_subset}-vox-${use_voxpopuli}-\
exp_dir=zipformer_audio_encoder/exp-96M-uniform-v2-out-ds-${output_downsampling_factor}-zipformer-\
ls-960-\
lr-batches-${lr_batches}-lr-hours-${lr_hours}-${mask_mode}-mask-p-${mask_prob}-l-${mask_length}-\
cha-mask-p-${mask_channel_prob}-l-${mask_channel_length}-\
${mixing_mode}-mix-p-${mixing_prob}-min-snr-${min_snr}-p-noise-${p_noise}-min-snr-${min_noise_snr}\
wavlm-large-layer-21-libri-mvq-cb-${num_codebooks}-shar-md-${max_duration}-norm-fbank-${normalize_fbank}

# exp_dir=zipformer_audio_encoder/exp-316M-uniform-zipformer-lh-large-giga-xl-emo-0-voxpopuli-1-lr-epochs-1.0-w2v2-mask-prob-0.5-mask-len-10-channel-mask-prob-0.25-len-15-musan-1-min-snr-10-rir-0-wavlm-large-layer-21-libri-mvq-cb-16-shar-md-400-16-gpus
# exp_dir=zipformer_audio_encoder/exp-debug

echo $exp_dir

torchrun --nproc_per_node=4 --master_port 13143 \
  zipformer_audio_encoder/train_multi_KD3_shar_w2v2_mask.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iter 300000 \
    --use-shar $use_shar --shar-dir data-shar/data-shar-wavlm-large-layer-21-normalize-cb${num_codebooks}-hdf5 \
    --base-lr $lr --estimate-epoch $estimate_epoch --lr-hours $lr_hours --lr-batches $lr_batches \
    --use-fp16 1 \
    --exp-dir $exp_dir \
    --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --use-librispeech $use_librispeech --full-libri 1 --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-voxpopuli $use_voxpopuli \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --use-emotion-dataset $use_emotion_dataset --repeat-emo $repeat_emo \
    --do-mvq 1 --mvq-loss-by-task $mvq_loss_by_task \
    --do-audio-tagging $do_audio_tagging --at-KD $at_KD \
    --enable-rir $enable_rir --rir-cuts None \
    --enable-musan $enable_musan \
    --batch-mixing $batch_mixing --batch-mixing-mode $mixing_mode --p-noise $p_noise --min-noise-snr $min_noise_snr\
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
    --manifest-dir data/vq_whisper_turbo_zh_en_16_v2 \
    --normalize-fbank $normalize_fbank \
    --spec-aug-time-warp-factor -1 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --bucketing-sampler $bucket_sampler --num-buckets $num_buckets --merge-buckets $merge_buckets --sync-buckets $sync_buckets \
    --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 6