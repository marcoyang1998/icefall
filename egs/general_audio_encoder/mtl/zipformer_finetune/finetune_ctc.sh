#!/usr/bin/env bash

cd /mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl
echo "Current dir: $PWD"

####### Mount the necessary disks #######
bash mount_brainllm_h.sh
ls -lh download/LibriSpeech
#########################################

############## PYTHON env ###############
source /home/housiyuan/miniconda3/etc/profile.d/conda.sh && conda activate encoder

work_dir=/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl
cd $work_dir

echo "Current Directory: $PWD"

export PYTHONPATH=./../../../:$PYTHONPATH
export PYTHONPATH=/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/lhotse_dev:$PYTHONPATH
#########################################

# data related
use_librispeech=1
full_libri=0

causal=0
lr=0.045

# finetune checkpoint
do_finetune=1
finetune_ckpt=zipformer_audio_encoder/exp-600M-uniform-v2-zipformer-out-ds-2-lh-large-giga-xl-voxpopuli-1-as-full-x2-all-audio-w2v2-mask-p-0.65-l-10-cha-mask-p-0.25-l-20-musan-p-0.5-min-snr-10-multi-mvq-wavlm-all-wavlm-large-cb16-1.0-dasheng-cb8-0.1-md300/iter-448000-avg-2.pt

output_ds=2
post_output_ds=1

warmup_batches=2000
warmup_start=0.0

freeze_encoder=0
freeze_encoder_steps=2000
# freeze_encoder=1
# freeze_encoder_steps=-1
encoder_lr_scale=0.05

md=500

exp_dir=zipformer_finetune/exp-finetune-600m-ctc-sap-4-gpus-warmup-${warmup_batches}-start-${warmup_start}-md-${md}-fp32-no-filter-rerun
# exp_dir=zipformer_finetune/exp-debug

echo $exp_dir

torchrun --nproc_per_node=8 --master_port=19290 \
  zipformer_finetune/finetune_asr_sap.py \
    --num-epochs 50 \
    --use-fp16 0 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank \
    --lang-dir data/lang_char \
    --base-lr $lr --warmup-batches $warmup_batches --warmup-start $warmup_start \
    --do-finetune $do_finetune --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,3,4,1,1,1 \
    --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
    --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
    --encoder-unmasked-dim 768,768,768,768,768,768,768 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --output-downsampling-factor $output_ds \
    --post-encoder-downsampling-factor $post_output_ds \
    --on-the-fly-feats 1 \
    --max-duration $md

for m in ctc-decoding; do
    for epoch in 30; do
        for avg in $(seq 18 -1 13); do
            python zipformer_finetune/decode_ctc_sap.py \
                --epoch $epoch \
                --avg $avg \
                --manifest-dir data/fbank \
                --lang-dir data/lang_char \
                --use-averaged-model 1 \
                --downsampling-factor 1,2,4,8,4,2,1 \
                --num-encoder-layers 1,2,3,4,1,1,1 \
                --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
                --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
                --encoder-unmasked-dim 768,768,768,768,768,768,768 \
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