#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriSpeech
#      You can find BOOKS.TXT, test-clean, train-clean-360, etc, inside it.
#      You can download them from https://www.openslr.org/12
#
#  - $dl_dir/lm
#      This directory contains the following files downloaded from
#       http://www.openslr.org/resources/11
#
#        - 3-gram.pruned.1e-7.arpa.gz
#        - 3-gram.pruned.1e-7.arpa
#        - 4-gram.arpa.gz
#        - 4-gram.arpa
#        - librispeech-vocab.txt
#        - librispeech-lexicon.txt
#        - librispeech-lm-norm.txt.gz
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  # 5000
  # 2000
  # 1000
  500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: Download LM"
  mkdir -p $dl_dir/lm
  if [ ! -e $dl_dir/lm/.done ]; then
    ./local/download_lm.py --out-dir=$dl_dir/lm
    touch $dl_dir/lm/.done
  fi
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LibriSpeech,
  # you can create a symlink
  #
  #   ln -sfv /path/to/LibriSpeech $dl_dir/LibriSpeech
  #
  if [ ! -d $dl_dir/LibriSpeech/train-other-500 ]; then
    lhotse download librispeech --full $dl_dir
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LibriSpeech manifest"
  # We assume that you have downloaded the LibriSpeech corpus
  # to $dl_dir/LibriSpeech
  mkdir -p data/manifests
  if [ ! -e data/manifests/.librispeech.done ]; then
    lhotse prepare librispeech -j $nj $dl_dir/LibriSpeech data/manifests
    touch data/manifests/.librispeech.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to $dl_dir/musan
  mkdir -p data/manifests
  if [ ! -e data/manifests/.musan.done ]; then
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for librispeech"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.librispeech.done ]; then
    ./local/compute_fbank_librispeech.py --perturb-speed False \
    touch data/fbank/.librispeech.done
  fi

  if [ ! -f data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz ]; then
    cat <(gunzip -c data/fbank/librispeech_cuts_train-clean-100.jsonl.gz) \
      <(gunzip -c data/fbank/librispeech_cuts_train-clean-360.jsonl.gz) \
      <(gunzip -c data/fbank/librispeech_cuts_train-other-500.jsonl.gz) | \
      shuf | gzip -c > data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz
  fi

  if [ ! -e data/fbank/.librispeech-validated.done ]; then
    log "Validating data/fbank for LibriSpeech"
    parts=(
      train-clean-100
      train-clean-360
      train-other-500
      test-clean
      test-other
      dev-clean
      dev-other
    )
    for part in ${parts[@]}; do
      python3 ./local/validate_manifest.py \
        data/fbank/librispeech_cuts_${part}.jsonl.gz
    done
    touch data/fbank/.librispeech-validated.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.musan.done ]; then
    ./local/compute_fbank_musan.py
    touch data/fbank/.musan.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/lm/librispeech-lexicon.txt |
    sort | uniq > $lang_dir/lexicon.txt

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi

  if [ ! -f $lang_dir/L.fst ]; then
    log "Converting L.pt to L.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L.pt \
      $lang_dir/L.fst
  fi

  if [ ! -f $lang_dir/L_disambig.fst ]; then
    log "Converting L_disambig.pt to L_disambig.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L_disambig.pt \
      $lang_dir/L_disambig.fst
  fi
fi


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir
    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp data/lang_phone/words.txt $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      log "Generate data for BPE training"
      files=$(
        find "$dl_dir/LibriSpeech/train-clean-100" -name "*.trans.txt"
        find "$dl_dir/LibriSpeech/train-clean-360" -name "*.trans.txt"
        find "$dl_dir/LibriSpeech/train-other-500" -name "*.trans.txt"
      )
      for f in ${files[@]}; do
        cat $f | cut -d " " -f 2-
      done > $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/bpe.model ]; then
      ./local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir

      log "Validating $lang_dir/lexicon.txt"
      ./local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bpe.model
    fi

    if [ ! -f $lang_dir/L.fst ]; then
      log "Converting L.pt to L.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L.pt \
        $lang_dir/L.fst
    fi

    if [ ! -f $lang_dir/L_disambig.fst ]; then
      log "Converting L_disambig.pt to L_disambig.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L_disambig.pt \
        $lang_dir/L_disambig.fst
    fi
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare bigram token-level P for MMI training"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/transcript_tokens.txt ]; then
      ./local/convert_transcript_words_to_tokens.py \
        --lexicon $lang_dir/lexicon.txt \
        --transcript $lang_dir/transcript_words.txt \
        --oov "<UNK>" \
        > $lang_dir/transcript_tokens.txt
    fi

    if [ ! -f $lang_dir/P.arpa ]; then
      ./shared/make_kn_lm.py \
        -ngram-order 2 \
        -text $lang_dir/transcript_tokens.txt \
        -lm $lang_dir/P.arpa
    fi

    if [ ! -f $lang_dir/P.fst.txt ]; then
      python3 -m kaldilm \
        --read-symbol-table="$lang_dir/tokens.txt" \
        --disambig-symbol='#0' \
        --max-order=2 \
        $lang_dir/P.arpa > $lang_dir/P.fst.txt
    fi
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $dl_dir/lm/3-gram.pruned.1e-7.arpa > data/lm/G_3_gram.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram.fst.txt ]; then
    # It is used for LM rescoring
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $dl_dir/lm/4-gram.arpa > data/lm/G_4_gram.fst.txt
  fi

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/HL.fst ]; then
      ./local/prepare_lang_fst.py  \
        --lang-dir $lang_dir \
        --ngram-G ./data/lm/G_3_gram.fst.txt
    fi
  done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compile HLG"
  ./local/compile_hlg.py --lang-dir data/lang_phone

  # Note If ./local/compile_hlg.py throws OOM,
  # please switch to the following command
  #
  # ./local/compile_hlg_using_openfst.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_hlg.py --lang-dir $lang_dir

    # Note If ./local/compile_hlg.py throws OOM,
    # please switch to the following command
    #
    # ./local/compile_hlg_using_openfst.py --lang-dir $lang_dir
  done
fi

# Compile LG for RNN-T fast_beam_search decoding
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Compile LG"
  ./local/compile_lg.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_lg.py --lang-dir $lang_dir
  done
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  log "Stage 11: Generate LM training data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    lang_dir=data/lang_bpe_${vocab_size}
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --lm-data $dl_dir/lm/librispeech-lm-norm.txt \
      --lm-archive $out_dir/lm_data.pt
  done
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
  log "Stage 12: Generate LM validation data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    if [ ! -f $out_dir/valid.txt ]; then
      files=$(
        find "$dl_dir/LibriSpeech/dev-clean" -name "*.trans.txt"
        find "$dl_dir/LibriSpeech/dev-other" -name "*.trans.txt"
      )
      for f in ${files[@]}; do
        cat $f | cut -d " " -f 2-
      done > $out_dir/valid.txt
    fi

    lang_dir=data/lang_bpe_${vocab_size}
    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --lm-data $out_dir/valid.txt \
      --lm-archive $out_dir/lm_data-valid.pt
  done
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
  log "Stage 13: Generate LM test data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    if [ ! -f $out_dir/test.txt ]; then
      files=$(
        find "$dl_dir/LibriSpeech/test-clean" -name "*.trans.txt"
        find "$dl_dir/LibriSpeech/test-other" -name "*.trans.txt"
      )
      for f in ${files[@]}; do
        cat $f | cut -d " " -f 2-
      done > $out_dir/test.txt
    fi

    lang_dir=data/lang_bpe_${vocab_size}
    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --lm-data $out_dir/test.txt \
      --lm-archive $out_dir/lm_data-test.pt
  done
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
  log "Stage 14: Sort LM training data"
  # Sort LM training data by sentence length in descending order
  # for ease of training.
  #
  # Sentence length equals to the number of BPE tokens
  # in a sentence.

  for vocab_size in ${vocab_sizes[@]}; do
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir
    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data.pt \
      --out-lm-data $out_dir/sorted_lm_data.pt \
      --out-statistics $out_dir/statistics.txt

    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data-valid.pt \
      --out-lm-data $out_dir/sorted_lm_data-valid.pt \
      --out-statistics $out_dir/statistics-valid.txt

    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data-test.pt \
      --out-lm-data $out_dir/sorted_lm_data-test.pt \
      --out-statistics $out_dir/statistics-test.txt
  done
fi
