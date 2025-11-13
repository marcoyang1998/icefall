#!/usr/bin/env bash

echo "Num gpus: $GPU_COUNT"

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

# the dasheng model version, base/medium/large
model_version=medium
full_libri=1

# full finetune
freeze_encoder=0
set_eval=0


torchrun --nproc_per_node=1 --master_port=19293 \
    dasheng/finetune_asr.py \
        --num-epochs 30 \
        --use-fp16 1 \
        --start-epoch 1 \
        --manifest-dir data/librispeech_manifest \
        --full-libri $full_libri \
        --model-version $model_version \
        --exp-dir dasheng/exp-finetune-asr-full-libri-${full_libri}-${model_version} \
        --max-duration 200

