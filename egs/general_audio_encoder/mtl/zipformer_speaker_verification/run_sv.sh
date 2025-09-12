#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/cpfs02/user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH

# data related
vox_subset=vox2
num_speakers=17982
speed_perturbation=1
truncated_voxceleb=0

lr=0.001
opt=adam
warmup_start=0.0
warmup_batches=5000
lr_epochs=3.5
lr_batches=7500

enable_musan=1
musan_prob=0.6
enable_rir=1
rir_prob=0.5

speaker_embed_dim=512
use_weighted_sum=1
output_ds=1
post_output_ds=1

# softmax related
use_aam=1
margin=0.2
scale=30

freeze_encoder=0
freeze_encoder_epochs=6
encoder_lr_scale=0.1

md=500

finetune_ckpt=zipformer_audio_encoder/exp-316M-uniform-v2-out-ds-1-zipformer-lh-large-giga-xl-emo-0-voxpopuli-1-lr-batches-7500-lr-hours-75000-w2v2-mask-p-0.5-l-10-cha-mask-p-0.25-l-15-batch-mix-p-0.3-min-snr--5-p-noise-0.7-min-snr-5-wavlm-large-layer-21-libri-mvq-cb-16-shar-md-400-16-gpus/iter-500000-avg-4.pt

exp_dir=zipformer_speaker_verification/exp-316M-uniform-v2-out-ds-1-zipformer-\
weighted-sum-${use_weighted_sum}-${opt}-lr-${lr}-lr-batches-${lr_batches}-\
use-aam-${use_aam}-margin-${margin}-scale-${scale}-\
${vox_subset}-trunc-${truncated_voxceleb}-\
musan-${enable_musan}-p-${musan_prob}-rir-${enable_rir}-p-${rir_prob}-sp-${speed_perturbation}-\
freeze-${freeze_encoder_epochs}-epochs-lr-scale-${encoder_lr_scale}-md-${md}

# exp_dir=zipformer_speaker_verification/exp-debug

torchrun --nproc_per_node=4 --master_port 13161 \
  zipformer_speaker_verification/train_sv.py \
    --num-epochs 20 \
    --start-epoch 1 \
    --use-fp16 1 \
    --manifest-dir data/fbank_voxceleb \
    --exp-dir $exp_dir \
    --opt $opt --base-lr $lr \
    --warmup-batches $warmup_batches --warmup-start $warmup_start \
    --lr-epochs $lr_epochs --lr-batches $lr_batches \
    --voxceleb-subset $vox_subset --truncated-voxceleb $truncated_voxceleb \
    --do-finetune 1 --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-epochs $freeze_encoder_epochs \
    --encoder-lr-scale $encoder_lr_scale \
    --causal 0 \
    --speed-perturbation $speed_perturbation \
    --use-weighted-sum $use_weighted_sum \
    --use-aam $use_aam --margin $margin --scale $scale \
    --num-speakers $num_speakers --speaker-embed-dim $speaker_embed_dim \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,2,3,1,1,1 \
    --feedforward-dim 3072,3072,3072,3072,3072,3072,3072 \
    --encoder-dim 1024,1024,1024,1024,1024,1024,1024 \
    --encoder-unmasked-dim 512,512,512,512,512,512,512 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --output-downsampling-factor $output_ds \
    --post-encoder-downsampling-factor $post_output_ds \
    --num-workers 4 \
    --enable-musan $enable_musan --musan-prob $musan_prob \
    --enable-rir $enable_rir --rir-prob $rir_prob \
    --on-the-fly-feats 1 \
    --max-duration $md 