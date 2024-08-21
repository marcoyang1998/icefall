#!/usr/bin/env bash


export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/xy/mnt/yangxiaoyu/workspace/icefall_multi_KD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="4,5,6,7"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

base_lr=0.045
max_duration=1000

do_finetune=1

# finetune_ckpt=multi_KD/exp_causal1_delta6KD_LS1_5fold+wenetspech0_0fold+as_unbalanced1+vox_1_vox2_base_lr_0.045_use_beats_1_scale_1.0_use_ecapa_1_layer_2_scale_10.0_1_scale_1.0_specaug0_musan0_with_task_ID_stop_early1_share_asr1_md1500_amp_bf16/iter-468000-avg-3.pt
finetune_ckpt=multi_KD/exp_causal1_delta6KD_LS1_5fold+wenetspech0_0fold+as_unbalanced1+vox_1_vox2_base_lr_0.045_use_beats_1_scale_1.0_use_ecapa_1_layer_2_scale_10.0_1_scale_1.0_specaug0_musan0_with_task_ID_stop_early1_share_asr1_md1500_amp_bf16/iter-468000-avg-5.pt

encoder_lr_scale=0.2
ecapa_lr_scale=0.2
beats_lr_scale=0.2
freeze_encoder=0
freeze_encoder_steps=12000
freezing_encoder_layer_idx="0,1,2"
freeze_modules="encoder_embed,encoder.encoders.0,encoder.encoders.1,encoder.encoders.2,ecapa_asp,ecapa_linear"
# freezing_encoder_layer_idx="-1"
# freeze_modules="None"
sync_other_tasks=1

use_encoder_projection=1
encoder_projection_dim=2560

use_librispeech=1
full_libri=1
repeat_librispeech=6

use_wenetspeech=0
repeat_wenetspeech=1

do_audio_tagging=1
audio_tagging_KD=1
beats_label=1
audioset_subset=unbalanced
audio_tagging_loss_scale=2.0

do_sv=0
use_voxceleb=0 # skip the vox cuts
voxceleb_subset="only_vox2"
speaker_input_idx=2
sv_loss_scale=10.0

causal=1
enable_musan=0

exp_dir=exp_finetune_asr_libri${use_librispeech}x${repeat_librispeech}_do_AT${do_audio_tagging}_${audioset_subset}_KD_scale${audio_tagging_loss_scale}_do_SV${do_sv}_${voxceleb_subset}_scale${sv_loss_scale}_freeze_${freeze_encoder_steps}steps_encoder_lr_scale${encoder_lr_scale}_freeze_3layers_ecapa_lr_scale${ecapa_lr_scale}_init_3_tasks_delta6_pretrain_avg_musan${enable_musan}_sync_task_md${max_duration}_amp_bf16_further

python multi_KD/train_multi_task.py \
    --world-size 4 \
    --num-epochs 20 \
    --base-lr $base_lr \
    --causal $causal \
    --start-epoch 1 \
    --use-fp16 0 --use-bf16 1 \
    --num-workers 2 \
    --inf-check 0 \
    --use-bpe 1 \
    --beats-label $beats_label \
    --manifest-dir data/fbank_LSVoxAs_with_whisper_large-v3_with_taskID \
    --exp-dir multi_KD/$exp_dir \
    --use-librispeech $use_librispeech --full-libri $full_libri  --repeat-librispeech $repeat_librispeech \
    --use-wenetspeech $use_wenetspeech --repeat-wenetspeech $repeat_wenetspeech \
    --max-duration $max_duration \
    --bucketing-sampler False \
    --do-finetune $do_finetune --finetune-ckpt $finetune_ckpt \
    --use-encoder-projection $use_encoder_projection --encoder-projection-dim $encoder_projection_dim \
    --freeze-modules $freeze_modules \
    --freeze-encoder $freeze_encoder --encoder-lr-scale $encoder_lr_scale \
    --ecapa-lr-scale $ecapa_lr_scale --beats-lr-scale $beats_lr_scale \
    --sync-other-tasks $sync_other_tasks \
    --freeze-encoder-steps $freeze_encoder_steps --freezing-encoder-layer-index $freezing_encoder_layer_idx \
    --do-audio-tagging $do_audio_tagging --audioset-subset $audioset_subset --audio-tagging-loss-scale $audio_tagging_loss_scale \
    --audio-tagging-KD $audio_tagging_KD \
    --do-sv $do_sv --voxceleb-subset $voxceleb_subset --sv-loss-scale $sv_loss_scale \
    --use-voxceleb $use_voxceleb \
    --speaker-input-idx $speaker_input_idx \
    --audioset-subset $audioset_subset \
    --use-task-id True \
    --enable-musan $enable_musan \
    --master-port 13495
    