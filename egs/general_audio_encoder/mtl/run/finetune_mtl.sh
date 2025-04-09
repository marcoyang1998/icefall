#!/usr/bin/env bash

source ~/anaconda3/bin/activate && conda activate encoder
source /mnt/cache/share_data/housiyuan/softwares/activate-cuda-11.8.sh

export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH

master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# data related
use_librispeech=1
full_libri=0
use_gigaspeech=0
gigaspeech_subset=s

causal=1
lr=0.045

do_audio_tagging=0
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=zipformer_audio_encoder/exp-lr-0.04-causal-ls-giga-xl-lh-large-mls-1-extra-zh-en-use-weread-1-as-full-multi-mvq-kd-at-kd-scale-5.0-whisper-all-firered-zh-bucket-sampler-md-320-fix/iter-300000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=2000
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.05

md=1000

exp_dir=zipformer_audio_encoder_finetune/exp-finetune-ls-100h-\
lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-xlarge-all-data-with-musan-rir-300k-fix

export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node 2 --nnodes 1 --node_rank $SLURM_PROCID --master_addr $master_addr \
  zipformer_audio_encoder/finetune_mtl.py \
    --num-epochs 30 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --exp-dir $exp_dir \
    --manifest-dir data_s3/fbank_librispeech \
    --base-lr $lr \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune 1 --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --chunk-size 8,32,64 \
    --left-context-frames 128,256 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --on-the-fly-feats 1 \
    --num-workers 20 \
    --max-duration $md

echo "Done"