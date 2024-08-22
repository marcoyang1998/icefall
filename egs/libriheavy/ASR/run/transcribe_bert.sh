#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="0"


exp_dir=exp_medium_BERT_text_encoder_memory_drop_0.05_md1000_with_style_1_with_context_list_0_pre_text_random_style_random_2_transforms

epoch=50
avg=10
use_pre_text=1

for method in greedy_search modified_beam_search; do
    for book in "test_clean" "test_other"; do
        for use_gt_pre_text in 0; do
            for num_history in $(seq 8 -1 0); do
                python ./zipformer_prompt_asr/transcribe_bert.py \
                    --epoch $epoch \
                    --avg $avg \
                    --causal False \
                    --exp-dir ./zipformer_prompt_asr/${exp_dir} \
                    --bpe-model  /star-data/xiaoyu/icefall_prompt_asr/egs/libriheavy/ASR/data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
                    --manifest-dir data/long_audio_${book}/long_audio_${book}.jsonl.gz \
                    --pre-text-transform mixed-punc \
                    --style-text-transform mixed-punc \
                    --method $method \
                    --beam-size 4 \
                    --num-history $num_history \
                    --use-pre-text $use_pre_text \
                    --use-gt-pre-text $use_gt_pre_text
            done
        done
    done
done