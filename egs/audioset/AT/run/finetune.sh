#!/usr/bin/env bash

. /star-xy/miniconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate k2_cuda11

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_audio_ssl:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="6,7"

echo $CUDA_VISIBLE_DEVICES

subset=balanced

do_finetune=1
init_modules=encoder,encoder_embed
finetune_ckpt=zipformer_pretraining/exp_full_mask_fbank_only_masked_frames_mask_prob0.65_specaug0_musan0_noise0.1/pretrained-30-avg-5.pt

lr_batches=100000
lr_epochs=100.0
classifier_lr_scale=10.0

python zipformer/finetune.py \
    --world-size 2 \
    --audioset-subset $subset \
    --num-epochs 25 \
    --start-epoch 1 \
    --exp-dir zipformer/exp_finetune_${subset}_classifier_lr_scale_${classifier_lr_scale}_from_only_masked_fbank_noise0.1_no_specaug_further_lr_epochs${lr_epochs}_lr_batches${lr_batches}_fix_rand \
    --use-fp16 1 \
    --do-finetune $do_finetune \
    --init-modules $init_modules \
    --finetune-ckpt $finetune_ckpt \
    --classifier-lr-scale $classifier_lr_scale \
    --lr-epochs $lr_epochs \
    --lr-batches $lr_batches \
    --max-duration 1000 \
    --master-port 13846
