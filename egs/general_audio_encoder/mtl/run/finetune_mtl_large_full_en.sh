#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH

# data related
use_librispeech=1
full_libri=1
use_gigaspeech=1
gigaspeech_subset=xl
use_libriheavy=1
libriheavy_subset=large
use_wenetspeech=False
wenetspeech_subset=L
use_audioset=0
repeat_audioset=4
audioset_subset=full
audio_tagging_loss_scale=4.0

causal=1
lr=0.045

do_audio_tagging=0
at_KD=1 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=zipformer_audio_encoder/exp-large-full-en-zh-audio-multi-kd-time-mask-ratio-1.5-shar/iter-400000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=6000
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.05

zip_sampler=0
bucket_sampler=1
md=300

exp_dir=zipformer_audio_encoder_finetune/exp-large-finetune-full-en-\
-lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-shar-400k
# exp_dir=zipformer_audio_encoder_finetune/exp-debug

torchrun \
  --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
  --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
  zipformer_audio_encoder/finetune_mtl.py \
    --num-epochs 2 \
    --max-iters 500000 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-libriheavy $use_libriheavy --libriheavy-subset $libriheavy_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --exp-dir $exp_dir \
    --manifest-dir data/vq_whisper_turbo_zh_en_16_v2 \
    --base-lr $lr \
    --do-audio-tagging $do_audio_tagging --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --use-audioset $use_audioset --audioset-subset $audioset_subset --repeat-audioset $repeat_audioset \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune 1 --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --chunk-size 8,32,64 \
    --left-context-frames 128,256 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --zip-sampler $zip_sampler --bucketing-sampler $bucket_sampler \
    --use-shar 1 --shar-dir data-shar-no-feat \
    --on-the-fly-feats 1 \
    --max-duration $md

echo "Done"