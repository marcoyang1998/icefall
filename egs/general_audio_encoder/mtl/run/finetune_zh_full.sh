#!/usr/bin/env bash

export PYTHONPATH=/fs-computility/INTERN6/housiyuan/xiaoyu/workspace/icefall_general_encoder:$PYTHONPATH

# data related
use_librispeech=0
full_libri=0
use_gigaspeech=0
gigaspeech_subset=s
use_wenetspeech=1
wenetspeech_subset=L

causal=1
lr=0.045

do_audio_tagging=0
at_KD=0 # need to set this to 0 for efficiency
mvq_KD=0

finetune_ckpt=zipformer_audio_encoder/exp-xlarge-lr-0.04-full-en-zh-audio-multi-kd-time-mask-ratio-2.0-shar/iter-400000-avg-4.pt

freeze_encoder=0
freeze_encoder_steps=6000
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.05

md=300

exp_dir=zipformer_audio_encoder_finetune/exp-finetune-wenet-${wenetspeech_subset}-\
lr-${lr}-causal-${causal}-freeze-encoder-${freeze_encoder}\
-freeze-${freeze_encoder_steps}-step-encoder-lr-scale-${encoder_lr_scale}\
-from-xlarge-shar-lr-0.04-400k


torchrun \
  --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
  --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
  zipformer_audio_encoder/finetune_mtl.py \
    --num-epochs 30 \
    --use-fp16 1 \
    --start-epoch 1 \
    --max-iters 200000 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --use-gigaspeech $use_gigaspeech --gigaspeech-subset $gigaspeech_subset \
    --use-wenetspeech $use_wenetspeech --wenetspeech-subset $wenetspeech_subset \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank_mtl \
    --base-lr $lr \
    --do-audio-tagging $do_audio_tagging \
    --mvq-KD $mvq_KD --at-KD $at_KD \
    --do-finetune 1 --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --bpe-model data/lang_bbpe_2000/bbpe.model \
    --chunk-size 8,32,64 \
    --left-context-frames 128,256 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,1024,2048,3072,2048,1024 \
    --encoder-dim 192,384,768,1024,768,384 \
    --encoder-unmasked-dim 192,256,320,512,320,256 \
    --on-the-fly-feats 1 \
    --use-shar 1 --shar-dir data-shar-no-feat \
    --max-duration $md

echo "Done"