#!/usr/bin/env bash

source ~/anaconda3/bin/activate && conda activate encoder
source /mnt/cache/share_data/housiyuan/softwares/activate-cuda-11.8.sh

# export PYTHONPATH=/mnt/cache/share_data/housiyuan/icefall_audio_encoder:$PYTHONPATH
export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH

master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

lr=0.045

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
repeat_audioset=3
audioset_subset=full
use_mls=1
use_extra_chinese_dataset=0
use_extra_english_dataset=0
use_weread=0

# augmentation
enable_rir=1
enable_musan=1
enable_spec_aug=1
time_mask_ratio=1.0

# mvq KD
separate_mvq=1
output_downsampling_factor=2
num_codebooks="16,16"
delta="0,0"
frame_rate_ratio="2,1"
mvq_loss_scales="1.0,1.0"

# at KD
audio_tagging_loss_scale=5.0
at_KD=1

# data related
use_shar=1
zip_sampler=1
bucket_sampler=0
at_weighted_sampler=0
at_num_samples=400000
max_duration=600

exp_dir=zipformer_audio_encoder/exp-96M-zipformer-non-streaming-ls-\
lh-${libriheavy_subset}-giga-${gigaspeech_subset}-mls-${use_mls}-wenet-${wenetspeech_subset}-\
multi-mvq-out-ds-${output_downsampling_factor}-mask-ratio-${time_mask_ratio}-musan-${enable_musan}-rir-${enable_rir}\

# exp_dir=zipformer_audio_encoder/exp-debug

echo $exp_dir
echo $SLURM_PROCID
echo $master_addr

torchrun \
  --nproc_per_node 8 --nnodes 1 --node_rank $SLURM_PROCID --master_addr $master_addr\
  zipformer_audio_encoder/train_multi_KD3_shar_multi_mvq.py \
    --num-epochs 2 \
    --start-epoch 1 \
    --max-iter 300000 \
    --use-shar $use_shar --shar-dir data-shar/data-shar-whisper-cb16-firered-cb16 \
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
    --separate-mvq $separate_mvq \
    --enable-musan $enable_musan --enable-rir $enable_rir \
    --enable-spec-aug $enable_spec_aug --time-mask-ratio $time_mask_ratio \
    --output-downsampling-factor $output_downsampling_factor \
    --num-encoder-layers 2,2,3,4,3,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,448,768,448,192 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --causal 0 \
    --manifest-dir data/vq_whisper_turbo_cb16_firered_zh_en_cb32 \
    --spec-aug-time-warp-factor -1 \
    --num-codebooks $num_codebooks --distillation-delta $delta --teacher-frame-ratio $frame_rate_ratio \
    --mvq-loss-scales $mvq_loss_scales \
    --bucketing-sampler $bucket_sampler --zip-sampler $zip_sampler --at-weighted-sampler $at_weighted_sampler --at-num-samples $at_num_samples \
    --num-buckets 10 \
    --on-the-fly-feats 1 \
    --max-duration $max_duration \
    --num-workers 16