#!/usr/bin/env bash

source ~/anaconda3/bin/activate && conda activate encoder
source /mnt/cache/share_data/housiyuan/softwares/activate-cuda-11.8.sh

# export PYTHONPATH=/mnt/cache/share_data/housiyuan/icefall_audio_encoder:$PYTHONPATH
export PYTHONPATH=/mnt/cache/share_data/housiyuan/lhotse:$PYTHONPATH
export PYTHONPATH=./../../../:$PYTHONPATH

master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# do finetune
do_finetune=1

# data related
use_librispeech=1
full_libri=1
repeat_librispeech=2
use_gigaspeech=1
gigaspeech_subset=xl
use_libriheavy=1
libriheavy_subset=large
use_wenetspeech=1
repeat_wenetspeech=1
wenetspeech_subset=L
use_audioset=1
repeat_audioset=6
audioset_subset=full
audio_tagging_loss_scale=2.0

use_weread=1
use_mls=1
extra_zh=1
extra_en=1

causal=1
lr=0.02
enable_musan=1
enable_rir=1

do_audio_tagging=1
at_KD=1 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=zipformer_audio_encoder/exp-lr-0.04-causal-ls-giga-xl-lh-large-mls-1-extra-zh-en-use-weread-1-as-full-multi-mvq-kd-at-kd-scale-5.0-whisper-all-firered-zh-bucket-sampler-md-320-fix/iter-500000-avg-4.pt

# freeze_encoder=0
# freeze_encoder_steps=6000
freeze_encoder=0
freeze_encoder_steps=-1
encoder_lr_scale=0.1

zip_sampler=1
bucket_sampler=0
md=300

exp_dir=zipformer_audio_encoder_finetune/exp-xlarge-finetune-mtl-full-en-zh\
-lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-use-mls-${use_mls}-use-weread-${use_weread}-extra-zh-${extra_zh}-extra-en-${extra_en}\
-at-scale-${audio_tagging_loss_scale}-from-xlarge-lr-0.04-from-v1.0-500k-full-data-fix-lh\
-ft-with-musan-rir-fix-lh-bigger-decoder

# exp_dir=zipformer_audio_encoder_finetune/exp-debug-xlarge

# torchrun \
#   --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
#   --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
torchrun \
  --nproc_per_node 8 --nnodes 3 --node_rank $SLURM_PROCID --master_addr $master_addr\
  zipformer_audio_encoder/finetune_mtl_en_zh.py \
    --num-epochs 2 \
    --max-iters 220000 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri --repeat-librispeech $repeat_librispeech \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset --repeat-wenetspeech $repeat_wenetspeech \
    --use-mls $use_mls \
    --use-weread $use_weread \
    --use-extra-chinese-dataset $extra_zh  --use-extra-english-dataset $extra_en \
    --exp-dir $exp_dir \
    --use-shar 1 --shar-dir data-shar/data-shar-whisper-zh-en-cb16-v2 \
    --manifest-dir data/vq_whisper_turbo_zh_en_16_v2 \
    --bpe-model data/lang_bbpe_2000/bbpe.model \
    --base-lr $lr \
    --do-audio-tagging $do_audio_tagging --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune $do_finetune --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --keep-last-k 60 \
    --joiner-dim 768 --decoder-dim 768 \
    --enable-musan $enable_musan --enable-rir $enable_rir \
    --chunk-size 8,32,-1 \
    --left-context-frames 128,256,-1 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --zip-sampler $zip_sampler --bucketing-sampler $bucket_sampler \
    --num-buckets 10 \
    --on-the-fly-feats 1 \
    --num-workers 20 \
    --max-duration $md

echo "Done"