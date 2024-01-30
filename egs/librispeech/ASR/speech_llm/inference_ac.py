#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:
(1) greedy search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

(4) fast beam search (one best)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(5) fast beam search (nbest)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
"""


import argparse
import csv
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from lhotse.cut import Cut
import nltk
from asr_datamodule import LibriSpeechAsrDataModule

from train_ac import add_model_arguments, get_model, get_params
from caption_evaluation_tools.eval_metrics import evaluate_metrics
from tokenization_milm import MiTokenizer

from icefall import ContextGraph, LmScorer, NgramLm
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    make_pad_mask,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - sample
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding-method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding-method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )
    
    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--use-full-fp16",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--use-shallow-fusion",
        type=str2bool,
        default=False,
        help="""Use neural network LM for shallow fusion.
        If you want to use LODR, you will also need to set this to true
        """,
    )

    parser.add_argument(
        "--lm-type",
        type=str,
        default="rnn",
        help="Type of NN lm",
        choices=["rnn", "transformer"],
    )

    parser.add_argument(
        "--nnlm-scale",
        type=float,
        default=0.3,
        help="""The scale of the neural network LM
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    parser.add_argument(
        "--tokens-ngram",
        type=int,
        default=2,
        help="""The order of the ngram lm.
        """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        default=500,
        help="ID of the backoff symbol in the ngram LM",
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=2,
        help="""
        The bonus score of each token for the context biasing words/phrases.
        Used only when --decoding-method is modified_beam_search and
        modified_beam_search_LODR.
        """,
    )

    parser.add_argument(
        "--context-file",
        type=str,
        default="",
        help="""
        The path of the context biasing lists, one word/phrase each line
        Used only when --decoding-method is modified_beam_search and
        modified_beam_search_LODR.
        """,
    )
    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm=None,
    ngram_lm_scale: float = 0.0,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      LM:
        A neural network language model.
      ngram_lm:
        A ngram language model
      ngram_lm_scale:
        The scale for the ngram language model.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    batch_size = feature.size(0)

    supervisions = batch["supervisions"]
    texts = supervisions["text"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encode_audio(feature, feature_lens)
    encoder_out_lens += 2

    hyps = []

    for i in range(batch_size):
        prompt = torch.full((1, 0), sp.bos_token_id).to(device)
        
        # initial empty prefix
        prompt_lens = torch.full((1,), 0).int().to(device)
        prompt_embeddings = model.embed_tokens(prompt)

        # the audio with <soa> and <eoa> embedding
        audio_embeddings, audio_lens = model.concat_token_embedings(
            x=encoder_out[i, None],
            x_lens=encoder_out_lens[i, None],
            y=prompt_embeddings,
            y_lens=prompt_lens,
        )
        
        input_ids = torch.tensor([[sp.bos_token_id] * audio_lens.item()]).long().to(device)

        generation_kwargs = {
            "do_sample": params.do_sample,
            "input_ids": input_ids,
            "audio_embeddings": audio_embeddings,
            "audio_lens": audio_lens,
            "max_new_tokens": 300,
            "pad_token_id": sp.pad_token_id,
            "eos_token_id": sp.eos_token_id,
        }
        
        output = model.llm.generate(**generation_kwargs)
        hyp = output[0, audio_lens:]
        hyps.append(hyp.tolist())

    # import pdb; pdb.set_trace()
    hyps = [sp.decode(hyp[:-1]).split() for hyp in hyps] # remove the endoftext token
    
    return {params.decoding_method: hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm=None,
    ngram_lm_scale: float = 0.0,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        cuts = batch["supervisions"]["cut"]
        
        cut_ids = [cut.id for cut in cuts]
        audio_catpions = [c.supervisions[0].audio_captions.split(";;") for c in cuts]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
            context_graph=context_graph,
            word_table=word_table,
            batch=batch,
            LM=LM,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(audio_catpions)
            for cut_id, hyp_words, ref_captions in zip(cut_ids, hyps, audio_catpions):
                ref_captions = [t.split() for t in ref_captions]
                this_batch.append((cut_id, ref_captions, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(audio_catpions)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results

def compute_bleu(results):
    # compute the bleu score for a set of sentences, each having a list of references
    num_sentences = len(results)
    bleu_scores = []
    for cut_id, ref_captions, hyp_words in results:
        score = nltk.bleu(ref_captions, hyp_words)
        bleu_scores.append(score)
    avg_score = sum(bleu_scores)/num_sentences
    
    return avg_score

def write_results_to_csv(results, csv_path):
    # the hyp csv file should have the following fields
    # file_name, caption_predicted
    # the ref csv file should have the following fields
    # file_name, caption_reference_01, ..., caption_reference_05
    
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "caption_predicted"])
        for (file_id, ref, hyp) in results:
            w.writerow([file_id, " ".join(hyp)])
    
    return csv_path

def save_results(
    params: AttributeDict,
    test_set_name: str,
    reference_csv: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.csv"
        )
        results = sorted(results)
        
        recog_path = write_results_to_csv(results, recog_path)
        logging.info(f"The generated results are stored in {recog_path}")
        
        metrics = evaluate_metrics(recog_path, reference_csv, nb_reference_captions=5)
        for m, scores in metrics.items():
            overall_score = scores["score"]
            logging.info(f"Evaluating on {test_set_name}, Metirc: {m}, score: {overall_score}")
        
        # logging.info("Storing the evaluation result for each testing clip")

@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "sample"
    )
    params.res_dir = params.exp_dir / params.decoding_method
    
    if params.decoding_method == "greedy_search":
        params.do_sample = False
    elif params.decoding_method == "sample":
        params.do_sample = True
    else:
        raise NotImplementedError(f"Unsupported decoding method {params.decoding_method}")

    if os.path.exists(params.context_file):
        params.has_contexts = True
    else:
        params.has_contexts = False

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"-chunk-{params.chunk_size}"
        params.suffix += f"-left-context-{params.left_context_frames}"

    if params.use_shallow_fusion:
        params.suffix += f"-{params.lm_type}-lm-scale-{params.lm_scale}"

        if "LODR" in params.decoding_method:
            params.suffix += (
                f"-LODR-{params.tokens_ngram}gram-scale-{params.ngram_lm_scale}"
            )

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = MiTokenizer(params.tokenizer_path, fix_zh=False)
    params.vocab_size = sp.vocab_size

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.load_state_dict(average_checkpoints(filenames, device="cpu"))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.load_state_dict(average_checkpoints(filenames, device="cpu"))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device="cpu",
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device="cpu",
                )
            )

    model.to(device)
    model.eval()

    # only load the neural network LM if required
    if params.use_shallow_fusion or params.decoding_method in (
        "modified_beam_search_lm_rescore",
        "modified_beam_search_lm_rescore_LODR",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_LODR",
    ):
        LM = LmScorer(
            lm_type=params.lm_type,
            params=params,
            device=device,
            lm_scale=params.lm_scale,
        )
        LM.to(device)
        LM.eval()
    else:
        LM = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    def add_dummy_text(c: Cut):
        if c.supervisions[0].text is None:
            c.supervisions[0].text = "Dummy text added as a place holder."
        return c

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    test_sets = []
    test_dls = []
    gt_captions = []
    
    if params.use_clotho:
        clotho_eval_cuts = librispeech.clotho_eval_cuts().map(add_dummy_text)
        clotho_test_dl = librispeech.test_dataloaders(clotho_eval_cuts)
        test_dls.append(clotho_test_dl)
        test_sets.append("eval_clotho")
        gt_captions.append("data/fbank_clotho/clotho_evaluation_captions.csv")
    if params.use_audiocaps:
        audiocaps_test_cuts = librispeech.audiocaps_test_cuts().map(add_dummy_text)
        audiocaps_test_dl = librispeech.test_dataloaders(audiocaps_test_cuts)
        test_dls.append(audiocaps_test_dl)
        test_sets.append("test_audiocaps")
        gt_captions.append("data/fbank_audiocaps/audiocaps_test_captions.csv")
    
    assert len(test_sets) >= 1

    for test_set, test_dl, gt_caption in zip(test_sets, test_dls, gt_captions):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            reference_csv=gt_caption,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
