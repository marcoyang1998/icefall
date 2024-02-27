#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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

import argparse
import copy
import logging
import warnings
from pathlib import Path
import random
from functools import partial
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse import load_manifest_lazy, CutSet
from lhotse.cut import Cut, MonoCut
from lhotse.dataset.collation import collate_custom_field
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import MultiKDModel
from model_speech_llm import SpeechLLMModel, WhisperEncoder
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer2

from modelscope import AutoTokenizer
from tokenization_qwen import QWenTokenizer
from modeling_qwen import QWenSpeechLLM

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    get_parameter_groups_with_lrs2,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]

def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    ) + 100000


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name

def get_tokenizer(params: AttributeDict):
    # sp = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/qwen/Qwen-1_8B",revision='master', trust_remote_code=True)
    sp = QWenTokenizer.from_pretrained("/root/.cache/modelscope/hub/qwen/Qwen-1_8B",revision='master', trust_remote_code=True)
    # deal with special tokens
    assert sp.decode(151643) == "<|endoftext|>"
    sp.eos_token = "<|endoftext|>"
    sp.eos_token_id = 151643
    
    assert sp.decode(151646) == "<|extra_0|>"
    sp.pad_token = "<|extra_0|>" # needed for batch padding
    sp.pad_token_id = 151646
    
    return sp

def get_task_prompt(
    task_name: str = "ASR",
    input_language: str = "en",
    output_language: str = "en",
) -> str:
    # Prepare the task specific prompt for the LLM
    task_tags = {
        "ASR": "<|transcribe|>",
        "AC": "<|audiocaption|>",
        "AST": "<|translate|>",
    }
    
    language_tags = {
        "en": "<|en|>",
        "zh": "<|zh|>",
        "de": "<|de|>",
        "es": "<|es|>",
        "ko": "<|ko|>",
        "fr": "<|fr|>",
        "ja": "<|ja|>",
        "it": "<|it|>",
        "unk": "<|unknown|>", # unknown languages
    }
    
    if task_name == "AC":
        input_language = "unk"
    
    return language_tags[input_language] + task_tags[task_name] + language_tags[output_language]

def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )
    
    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
    )
    
    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network)" "part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )
    
    parser.add_argument(
        "--multitask",
        type=str2bool,
        default=True,
        help="If perform multitask training"
    )
    
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="The length of a learnable prefix after the audio soft prompt"
    )
   
    parser.add_argument(
        "--freeze-embeddings",
        type=str2bool,
        default=True,
        help="If freezing the token embeddings in the LLM"
    )

    
    parser.add_argument(
        "--speech-encoder-path",
        type=str,
        default=None,
        help="The initialization of the speech encoder. Won't be used if not specified"
    )

    parser.add_argument(
        "--llm-embed-dim",
        type=int,
        default=1536 ,
        help="Dimension of the embedding for the language model",
    )
    
    parser.add_argument(
        "--do-avg-pooling",
        type=str2bool,
        default=False,
        help="If perform average pooling after the audio encoder to reduce frame rate"
    )
    
    parser.add_argument(
        "--pooling-stride",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--do-sv",
        type=str2bool,
        default=True,
        help="If do SV, this will add the speaker module to the model"
    )
    
    parser.add_argument(
        "--speaker-input-idx",
        type=int,
        default=-1,
        help="Which layer's output to be used for speaker embeddings"
    )

    parser.add_argument(
        "--do-audio-tagging",
        type=str2bool,
        default=False,
        help="If do audio tagging multi task training"
    )
    
    parser.add_argument(
        "--use-whisper-encoder",
        type=str2bool,
        default=False,
        help="If use the whisper encoder as the speech encoder"
    )
    
    parser.add_argument(
        "--whisper-version",
        type=str,
        default="small.en",
        help="The version of whisper to be used"
    )

    parser.add_argument(
        "--use-encoder-projection",
        type=str2bool,
        default=False,
        help="If add a final projection layer at the end of the encoder"
    )

    parser.add_argument(
        "--encoder-projection-dim",
        type=int,
        default=-1,
        help="The output dimension of the projection"
    )

    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--ecapa-lr-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--use-task-id",
        type=str2bool,
        default=False,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--warmup-start",
        type=float,
        default=0.5,
        help="The initial learning rate during warmup."
    )

    parser.add_argument(
        "--warmup-batches",
        type=float,
        default=500.0,
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=10,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )
    
    parser.add_argument(
        "--use-full-fp16",
        type=str2bool,
        default=False,
        help="Whether to run everything on fp16, including the speech encoder",
    )
    
    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use pure bf16 training.",
    )
    
    parser.add_argument(
        "--repeat-librispeech",
        type=int,
        default=1,
        help="How many times to repeat LS",
    )
    
    parser.add_argument(
        "--repeat-covost",
        type=int,
        default=1,
        help="How many times to repeat covost",
    )
    
    parser.add_argument(
        "--repeat-aishell",
        type=int,
        default=1,
        help="How many times to repeat aishell",
    )
    
    parser.add_argument(
        "--repeat-AC",
        type=int,
        default=3,
        help="How many times to repeat audio caption data",
    )

    parser.add_argument(
        "--use-lowercase",
        type=str2bool,
        default=False,
        help="Whether to lowercase the input",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - encoder_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 6000,  # For the 100h subset, use 800
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params

def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))

def get_llm_decoder(params: AttributeDict) -> nn.Module:
    # Load a pre-trained LLM decoder
    if params.use_bf16:
        decoder = QWenSpeechLLM.from_pretrained("qwen/Qwen-1_8B", revision='master', trust_remote_code=True, bf16=True).eval()
        logging.info("Loading LLM params in bf16 format")
    else:
        decoder = QWenSpeechLLM.from_pretrained("qwen/Qwen-1_8B", revision='master', trust_remote_code=True, fp16=True).eval()
        logging.info("Loading LLM params in fp16 format")
    
    if not params.use_full_fp16 and not params.use_bf16:
        decoder = decoder.to(torch.float32)
        logging.info("Convering the LLM parameter to fp32 format")
        
    # set requires_grad=False for all the parameters in llm
    if not params.freeze_embeddings:
        embedding_modules = ["transformer.wte", "lm_head"]
    else:
        embedding_modules = []
        
    for name, param in decoder.named_parameters():
        param.requires_grad = False
        for m in embedding_modules:
            if m in name:
                param.requires_grad = True
                logging.info(f"Don't freeze {name}")
    return decoder
    

def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder

def get_speech_encoder_model(params: AttributeDict) -> nn.Module:
    if params.use_whisper_encoder:
        model = WhisperEncoder(
            whisper_version=params.whisper_version,
        )
        params.encoder_dim = model.encoder_dim
    else:
        encoder_embed = get_encoder_embed(params)
        encoder = get_encoder_model(params)

        model = MultiKDModel(
            encoder_embed=encoder_embed,
            encoder=encoder,
            encoder_dim=max(_to_int_tuple(params.encoder_dim)),
            use_beats=False,
            use_ecapa=False,
            use_whisper=params.use_encoder_projection,
            whisper_dim=1280, # This is fixed as the encoder is pre-trained with whisper-large
            speaker_input_idx=params.speaker_input_idx,
            use_subsampled_output=True,
        )
    
        if params.speech_encoder_path is not None:
            logging.info(f"Initialising the speech encoder from {params.speech_encoder_path}")
            state_dict = torch.load(params.speech_encoder_path, map_location="cpu")["model"]
            keys = [k for k in state_dict.keys() if k.startswith("encoder") or k.startswith("encoder_embed")]
            state_dict = {k: state_dict[k] for k in keys}
            
            model.load_state_dict(state_dict, strict=False)
    
    return model

def get_model(params: AttributeDict) -> nn.Module:
    # if use whisper encoder, cannot set use-encoder-projection to True
    assert not (params.use_encoder_projection and params.use_whisper_encoder)
    
    speech_encoder = get_speech_encoder_model(params)
    llm_decoder = get_llm_decoder(params)
    
    if params.use_whisper_encoder:
        speech_encoder_dim = params.encoder_dim
    else:
        if params.use_encoder_projection:
            speech_encoder_dim = params.encoder_projection_dim
        else:
            speech_encoder_dim = max(_to_int_tuple(params.encoder_dim))
    
    model = SpeechLLMModel(
        llm=llm_decoder,
        llm_embed_dim=params.llm_embed_dim,
        vocab_size=params.vocab_size,
        speech_encoder=speech_encoder,
        speech_encoder_dim=speech_encoder_dim,
        do_avg_pooling=params.do_avg_pooling,
        pooling_stride=params.pooling_stride,
        prefix_len=params.prefix_len,
        multitask=params.multitask,
        pad_token=params.pad_token_id,
    )
    
    return model

def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)
    if params.use_bf16:
        feature = feature.to(torch.bfloat16)

    supervisions = batch["supervisions"]
    cuts = batch["supervisions"]["cut"]
    cut_ids = [c.id for c in cuts]
    task_names = [c.task_name for c in cuts]
    input_languages = [c.input_language for c in cuts]
    output_languages = [c.output_language for c in cuts]
        
    task_prompts = [
        get_task_prompt(
            task_name=task_name, input_language=in_language, output_language=out_language,
        ) for task_name, in_language, out_language in zip(task_names, input_languages, output_languages)
    ]
    if is_training and random.random() < 0.02:
        logging.info(f"Task prompt: {task_prompts}")
        logging.info(f"Cut IDs: {cut_ids}")

    feature_lens = supervisions["num_frames"].to(device)

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    texts = batch["supervisions"]["text"]
    if params.use_lowercase:
        texts = [s.lower() for s in texts]
    
    texts = [prompt + t + params.eos_token for prompt, t in zip(task_prompts, texts)]

    encoded_texts = sp.batch_encode_plus(texts, return_tensors="pt", return_length=True, padding=True).to(device) # Has EOS
    y = encoded_texts["input_ids"]
    y_lens = encoded_texts["length"]
    
    # This should be changed if we change the task prompt
    text_prompt_lens = torch.tensor([3] * len(texts), device=device).long() # no bos token is needed, thus 0

    with torch.set_grad_enabled(is_training):
        nll_loss = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            y_lens=y_lens,
            text_prompt_lens=text_prompt_lens
        )

        nan_mask = nll_loss.isnan()
        if torch.any(nan_mask):
            logging.info("Masking nan in loss values")
        
        nll_loss[nan_mask] = 0.0
        nll_loss = nll_loss.sum()

        loss = 0.0
        loss += nll_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
        info["tokens"] = (y_lens - 1).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    info["nll_loss"] = nll_loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dls: List[torch.utils.data.DataLoader],
    valid_sets: List[str],
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))
        
        if params.on_the_fly_feats:
            torch.cuda.empty_cache()
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16 and not params.use_bf16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            if params.use_bf16:
                loss.backward()
                scheduler.step_batch(params.batch_idx_train)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scheduler.step_batch(params.batch_idx_train)

                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            # save_bad_model()
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and (params.use_fp16 and not params.use_bf16):
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if (params.use_fp16 and not params.use_bf16) else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if (params.use_fp16 and not params.use_bf16) else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            for valid_set, valid_dl in zip(valid_sets, valid_dls):
                valid_info = compute_validation_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    valid_dl=valid_dl,
                    world_size=world_size,
                )
            
                logging.info(f"Epoch {params.cur_epoch}, validation on {valid_set}: {valid_info}")
                
                if tb_writer is not None:
                    valid_info.write_summary(
                        tb_writer, f"train/valid_{valid_set}", params.batch_idx_train
                    )
            logging.info(
                    f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
                )
            model.train() # set to train because model was set to eval when computing valid loss

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")
    
    sp = get_tokenizer(params)
    
    params.eos_token = sp.eos_token
    params.vocab_size = sp.vocab_size
    params.pad_token_id = sp.pad_token_id

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    num_trainable_param = sum([p.numel() * p.requires_grad for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    logging.info("Number of trainable model parameters: {} (percentage: {:.2f}%)".format(num_trainable_param, num_trainable_param/num_param * 100))

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)
        
    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if not params.use_full_fp16 and not params.use_bf16:
        model.to(torch.float32)
        logging.info("Converting full model to fp32")
    else:
        model.to(torch.bfloat16)
        logging.info("Converting full model to bf16")
        
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # only feed the parameters in the speech encoder to the optimizer
    parameters = get_parameter_groups_with_lrs2(
        model, lr=params.base_lr, include_names=True,
    )        

    optimizer = ScaledAdam(
        parameters,
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(
        optimizer, 
        params.lr_batches, 
        params.lr_epochs, 
        warmup_start=params.warmup_start,
        warmup_batches=params.warmup_batches,
    )

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    librispeech = LibriSpeechAsrDataModule(args)

    train_cuts = []
    sampling_weights = []
    
    def _set_task_prompt(task_name, input_language, output_language, c):
        c.task_name = task_name
        c.input_language = input_language
        c.output_language = output_language
        return c
    
    def remove_short_and_long_utt_covost(c: Cut):
        if c.duration < 1.0 or c.duration > 20.0:
            return False
        return True
    
    assert params.use_librispeech or params.use_clotho or params.use_audiocaps or params.use_covost
    
    # ASR data
    if params.use_librispeech:
        if not params.full_libri: 
            librispeech_cuts = librispeech.train_clean_100_cuts().repeat(
                times=params.repeat_librispeech,
                preserve_id=False,
            )
            librispeech_cuts_len = 28539 * 3  # with speed purturb
        else:
            librispeech_cuts = librispeech.train_all_shuf_cuts().repeat(
                times=params.repeat_librispeech,
                preserve_id=False,
            )
            librispeech_cuts_len = 281239 * 3 # with speed purturb
        
        librispeech_cuts = librispeech_cuts.map(partial(_set_task_prompt, "ASR", "en", "en"))
        train_cuts.append(librispeech_cuts)
        sampling_weights.append(librispeech_cuts_len)
        
    if params.use_aishell:
        aishell_cuts = librispeech.aishell_train_cuts().repeat(
            times=params.repeat_aishell,
            preserve_id=False,
        )
        aishell_cuts_len = 360294
        
        aishell_cuts = aishell_cuts.map(partial(_set_task_prompt, "ASR", "zh", "zh"))
        train_cuts.append(aishell_cuts)
        sampling_weights.append(aishell_cuts_len)

    def _set_translation_as_text(c: Cut):
        c.supervisions[0].text = c.translation
        return c
    
    # AST data
    if params.use_covost:
        covost_cuts = librispeech.covost_train_cuts().repeat(
            times=params.repeat_covost,
            preserve_id=False,
        )
        covost_cuts_len = 232939

        covost_cuts = covost_cuts.map(_set_translation_as_text)
        covost_cuts = covost_cuts.map(partial(_set_task_prompt, "AST", "en", "zh"))
        covost_cuts = covost_cuts.filter(remove_short_and_long_utt_covost)
        train_cuts.append(covost_cuts)
        sampling_weights.append(covost_cuts_len)

    # AC data
    if params.use_clotho:
        clotho_cuts = librispeech.clotho_train_cuts().repeat(
            params.repeat_AC,
            preserve_id=False,
        )
        clotho_cuts = clotho_cuts.map(partial(_set_task_prompt, "AC", "unk", "en"))
        train_cuts.append(clotho_cuts)
        sampling_weights.append(14465 * params.repeat_AC)
    
    if params.use_audiocaps:
        audiocaps_cuts = librispeech.audiocaps_train_cuts().repeat(
            params.repeat_AC,
            preserve_id=False,
        )
        audiocaps_cuts =audiocaps_cuts.map(partial(_set_task_prompt, "AC", "unk", "en"))
        train_cuts.append(audiocaps_cuts)
        sampling_weights.append(42104 * params.repeat_AC)

    if len(train_cuts) > 1:
        logging.info(f"Using mux to combine {train_cuts}")
        logging.info(f"Using weights: {sampling_weights}")
        train_cuts = CutSet.mux(
            *train_cuts,
            weights=sampling_weights,
            stop_early=True,
        )
    else:
        train_cuts = train_cuts[0]

    logging.info(train_cuts)

    def remove_short_and_long_utt(c: Cut):
        if c.duration < 1.0 or c.duration > 28.0:
            return False
        
        return True
    
    def add_dummy_text(c: Cut):
        if c.supervisions[0].text is None:
            assert c.supervisions[0].audio_captions is not None
            c.supervisions[0].text = c.supervisions[0].audio_captions
        return c

    train_cuts = train_cuts.map(add_dummy_text)
    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_dls = []
    valid_sets = []

    # For ASR
    if params.use_librispeech:
        valid_cuts = librispeech.dev_clean_cuts()
        valid_cuts += librispeech.dev_other_cuts()
        valid_cuts = valid_cuts.map(partial(_set_task_prompt, "ASR", "en", "en"))
        valid_cuts = valid_cuts.filter(remove_short_and_long_utt)
        valid_dl = librispeech.valid_dataloaders(valid_cuts)
        
        valid_dls.append(valid_dl)
        valid_sets.append("ASR")
        
    if params.use_aishell:
        aishell_valid_cuts = librispeech.aishell_dev_cuts()
        aishell_valid_cuts = aishell_valid_cuts.map(partial(_set_task_prompt, "ASR", "zh", "zh"))
        aishell_valid_cuts = aishell_valid_cuts.filter(remove_short_and_long_utt)
        aishell_valid_dl = librispeech.valid_dataloaders(aishell_valid_cuts)
        
        valid_dls.append(aishell_valid_dl)
        valid_sets.append("ASR_aishell")
        
    # For AST 
    if params.use_covost:
        covost_valid_cuts = librispeech.covost_dev_cuts()
        covost_valid_cuts = covost_valid_cuts.map(_set_translation_as_text)
        covost_valid_cuts = covost_valid_cuts.map(partial(_set_task_prompt, "AST", "en", "zh"))
        covost_valid_cuts = covost_valid_cuts.filter(remove_short_and_long_utt_covost)
        covost_valid_dl = librispeech.valid_dataloaders(covost_valid_cuts)
        
        valid_dls.append(covost_valid_dl)
        valid_sets.append("AST_covost")
    
    # For AC
    if params.use_clotho:
        clotho_valid_cuts = librispeech.clotho_eval_cuts()
        clotho_valid_cuts = clotho_valid_cuts.map(partial(_set_task_prompt, "AC", "unk", "en")).map(add_dummy_text)
        clotho_valid_dl = librispeech.valid_dataloaders(clotho_valid_cuts)
        
        valid_dls.append(clotho_valid_dl)
        valid_sets.append("AC_clotho")
    
    if params.use_audiocaps:
        audiocaps_valid_cuts = librispeech.audiocaps_val_cuts()
        audiocaps_valid_cuts = audiocaps_valid_cuts.map(partial(_set_task_prompt, "AC", "unk", "en")).map(add_dummy_text)
        audiocaps_valid_dl = librispeech.valid_dataloaders(audiocaps_valid_cuts)
        
        valid_dls.append(audiocaps_valid_dl)
        valid_sets.append("AC_audiocaps")
    
    logging.info(valid_sets)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dls=valid_dls,
            valid_sets=valid_sets,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    y = sp.batch_encode_plus(supervisions["text"])
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
