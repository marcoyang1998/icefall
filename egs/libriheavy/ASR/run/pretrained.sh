#!/usr/bin/env bash

. /star-xy/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate prompt_asr

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/star-data/xiaoyu/icefall_prompt_asr/:$PYTHONPATH
source new_env.sh

export CUDA_VISIBLE_DEVICES="5"

for use_pre_text in 0 1; do
    python ./zipformer_prompt_asr/pretrained.py \
    --checkpoint ./zipformer_prompt_asr/exp_medium_BERT_memory_layer_0_memory_drop_0.05_md1000_with_style_1_with_context_list_1_2_styles_fixed_upper_fixed_BERT_rerun/pretrained-epoch-50-avg-10.pt \
    --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
    --method modified_beam_search \
    --use-pre-text $use_pre_text \
    --style-text-transform mixed-punc \
    --style-prompt "Mixed-case English transcription, with punctuation." \
    --content-prompt "brahman" \
    --use-style-prompt True \
    brahman.flac
done