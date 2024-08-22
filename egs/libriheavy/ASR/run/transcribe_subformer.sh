#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="7"


exp_dir=exp_causal_0_medium_subformer_freeze_1_base_lr_0.045_same_bpe_with_context_list_0_more_distractors_2_styles_rerun

epoch=50
avg=10

use_pre_text=1

for method in modified_beam_search; do
    for book in "test_clean" "test_other"; do
        for use_gt_pre_text in 0; do
            for num_history in $(seq 8 -1 0); do
                python ./zipformer_prompt_asr/transcribe_subformer.py \
                    --epoch $epoch \
                    --avg $avg \
                    --causal False \
                    --exp-dir ./zipformer_prompt_asr/${exp_dir} \
                    --bpe-model /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/data/lang_bpe_500/bpe.model \
                    --text-encoder-bpe-model /star-xy/softwares/icefall_development/icefall_subformer_lm/egs/libriheavy/LM/data/lang_bpe_500/bpe.model \
                    --text-encoder-causal False \
                    --load-pretrained False \
                    --manifest-dir data/long_audio_${book}/long_audio_${book}.jsonl.gz \
                    --pre-text-transform mixed-punc \
                    --style-text-transform mixed-punc \
                    --method $method \
                    --num-history $num_history \
                    --use-pre-text $use_pre_text \
                    --use-gt-pre-text $use_gt_pre_text
            done
        done
    done
done