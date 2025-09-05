#!/usr/bin/env bash

export PYTHONPATH=./../../../:$PYTHONPATH

# data related
use_librispeech=1
full_libri=0

causal=0
lr=0.045

# finetune checkpoint
do_finetune=1
finetune_ckpt=zipformer_audio_encoder/exp-316M-uniform-v2-out-ds-1-zipformer-lh-large-giga-xl-emo-0-voxpopuli-1-lr-batches-7500-lr-hours-75000-w2v2-mask-p-0.5-l-10-cha-mask-p-0.25-l-15-batch-mix-p-0.3-min-snr--5-p-noise-0.7-min-snr-5-wavlm-large-layer-21-libri-mvq-cb-16-shar-md-400-16-gpus/iter-500000-avg-4.pt

output_ds=1
post_output_ds=1

freeze_encoder=0
freeze_encoder_steps=2000
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.05

md=1000

exp_dir=zipformer_finetune/exp-finetune-ctc

echo $exp_dir

torchrun --nproc_per_node=2 --master_port=19290 \
  zipformer_finetune/finetune_asr.py \
    --num-epochs 30 \
    --use-fp16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank \
    --lang-dir k2ssl-asr-librispeech-100h-zipformer-base-ctc/data/lang_char \
    --base-lr $lr \
    --do-finetune $do_finetune --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,2,3,1,1,1 \
    --feedforward-dim 3072,3072,3072,3072,3072,3072,3072 \
    --encoder-dim 1024,1024,1024,1024,1024,1024,1024 \
    --encoder-unmasked-dim 512,512,512,512,512,512,512 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --output-downsampling-factor $output_ds \
    --post-encoder-downsampling-factor $post_output_ds \
    --on-the-fly-feats 1 \
    --max-duration $md

for m in ctc-decoding; do
    for epoch in 30; do
        for avg in $(seq 18 -1 16); do
            python zipformer_finetune/decode_ctc.py \
                --epoch $epoch \
                --avg $avg \
                --manifest-dir data/fbank_librispeech \
                --lang-dir k2ssl-asr-librispeech-100h-zipformer-base-ctc/data/lang_char \
                --use-averaged-model 1 \
                --downsampling-factor 1,2,4,8,4,2,1 \
                --num-encoder-layers 1,2,2,3,1,1,1 \
                --feedforward-dim 3072,3072,3072,3072,3072,3072,3072 \
                --encoder-dim 1024,1024,1024,1024,1024,1024,1024 \
                --encoder-unmasked-dim 512,512,512,512,512,512,512 \
                --cnn-module-kernel 31,31,15,15,15,31,31 \
                --num-heads 8,8,8,8,8,8,8 \
                --output-downsampling-factor $output_ds \
                --post-encoder-downsampling-factor $post_output_ds \
                --on-the-fly-feats 1 \
                --exp-dir $exp_dir \
                --decoding-method $m \
                --max-duration 1000
        done
    done
done

# rm $exp_dir/*.pt

echo "Done"