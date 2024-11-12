#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
# run step 0 to step 5 by default
stage=0
stop_stage=5

# Note: This script just prepare the minimal requirements that needed by a
# transducer training with bpe units.
#
# If you want to use ngram or nnlm, please continue running prepare_lm.sh after
# you succeed running this script.
#
# This script also contains the steps to generate phone based units, but they
# will not run automatically, you can generate the phone based units by
# bash prepare.sh --stage -1 --stop-stage -1
# bash prepare.sh --stage 6 --stop-stage 6


# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriSpeech
#      You can find BOOKS.TXT, test-clean, train-clean-360, etc, inside it.
#      You can download them from https://www.openslr.org/12
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
#
# lm directory is not necessary for transducer training with bpe units, but it
# is needed by phone based modeling, you can download it by running
# bash prepare.sh --stage -1 --stop-stage -1
# then you can see the following files in the directory.
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

log "Running prepare.sh"

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
    ./local/compute_fbank_librispeech.py --perturb-speed False
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
  log "Stage 5: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir

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
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  if [ ! -f $dl_dir/lm/librispeech-lexicon.txt ]; then
    log "No lexicon file in $dl_dir/lm, please run :"
    log "prepare.sh --stage -1 --stop-stage -1"
    exit -1
  fi

  if [ ! -f $lang_dir/lexicon.txt ]; then
    (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
      cat - $dl_dir/lm/librispeech-lexicon.txt |
      sort | uniq > $lang_dir/lexicon.txt
  fi

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
