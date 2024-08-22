#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-xy/softwares/icefall_development/icefall_zipformer_prompt_asr_from_dan:$PYTHONPATH
source env.sh
export CUDA_VISIBLE_DEVICES="4"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

exp_dir=exp_baseline_small_causal_0_2gpus_md1000_mixed_trans

causal=0
long_audio_recog=1
use_ls_test_set=0

decoding_method=modified_beam_search

for epoch in 60; do
    for avg in 30; do
        for md in 1000; do
            python ./zipformer_prompt_asr/decode_baseline.py \
                --epoch $epoch \
                --avg $avg \
                --use-averaged-model True \
                --causal False \
                --exp-dir ./zipformer_prompt_asr/${exp_dir} \
                --causal $causal \
                --chunk-size 32 \
                --left-context-frames 256 \
                --manifest-dir data/fbank \
                --bpe-model ./data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
                --long-audio-recog $long_audio_recog \
                --long-audio-cuts data/fbank/libriheavy_cuts_medium.jsonl.gz \
                --max-duration $md \
                --compute-CER False \
                --use-ls-test-set $use_ls_test_set \
                --post-normalization 1 \
                --decoding-method $decoding_method
        done
    done
done