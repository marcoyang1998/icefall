#!/usr/bin/env bash

fbank_dir=$1
if [ ! -f $fbank_dir/librispeech_cuts_train-all-shuf.jsonl.gz ]; then
cat <(gunzip -c $fbank_dir/librispeech_cuts_train-clean-100.jsonl.gz) \
    <(gunzip -c $fbank_dir/librispeech_cuts_train-clean-360.jsonl.gz) \
    <(gunzip -c $fbank_dir/librispeech_cuts_train-other-500.jsonl.gz) | \
    shuf | gzip -c > $fbank_dir/librispeech_cuts_train-all-shuf.jsonl.gz
fi