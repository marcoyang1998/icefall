#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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

export CUDA_VISIBLE_DEVICES="0,1,2,3"


./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --audioset-subset full \
  --max-duration 1000


"""

import argparse
import copy
import csv
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from at_datamodule import AudioSetATDatamodule
from lhotse.cut import Cut, MonoCut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.dataset.collation import collate_custom_field
from lhotse.utils import fix_random_seed
from model import AudioTaggingModel
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer2

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
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]

beats_id2str = {0: '/m/078jl', 1: '/m/07rjwbb', 2: '/m/04rlf', 3: '/m/07qb_dv', 4: '/t/dd00125', 5: '/m/02sgy', 6: '/m/0342h', 7: '/m/042v_gx', 8: '/m/04szw', 9: '/m/0b_fwt', 10: '/m/0fx80y', 11: '/m/07rqsjt', 12: '/m/018vs', 13: '/m/07szfh9', 14: '/t/dd00128', 15: '/m/02fsn', 16: '/m/0d8_n', 17: '/m/0l14_3', 18: '/m/07q4ntr', 19: '/m/07rwj3x', 20: '/m/09x0r', 21: '/m/01s0vc', 22: '/m/025wky1', 23: '/m/07ryjzk', 24: '/m/07xzm', 25: '/m/01qbl', 26: '/m/026t6', 27: '/m/0283d', 28: '/m/02hnl', 29: '/m/02lkt', 30: '/m/03qtq', 31: '/m/06rvn', 32: '/m/0bm02', 33: '/m/0l14md', 34: '/m/03qc9zr', 35: '/g/122z_qxw', 36: '/m/019jd', 37: '/m/02rlv9', 38: '/m/07qcpgn', 39: '/m/011k_j', 40: '/m/0239kh', 41: '/m/03t3fj', 42: '/m/07r_80w', 43: '/m/09d5_', 44: '/m/01g50p', 45: '/m/04zmvq', 46: '/m/06d_3', 47: '/m/07jdr', 48: '/m/07pt_g0', 49: '/m/015lz1', 50: '/m/07sr1lc', 51: '/t/dd00126', 52: '/m/07rrlb6', 53: '/m/0fx9l', 54: '/m/07qmpdm', 55: '/t/dd00121', 56: '/m/01yg9g', 57: '/m/07qlf79', 58: '/m/01jnbd', 59: '/m/03fwl', 60: '/m/0ch8v', 61: '/m/0jbk', 62: '/m/07plz5l', 63: '/t/dd00036', 64: '/t/dd00003', 65: '/m/07lnk', 66: '/m/02fs_r', 67: '/m/02cz_7', 68: '/m/02dgv', 69: '/m/02y_763', 70: '/m/015y_n', 71: '/m/04fgwm', 72: '/m/030rvx', 73: '/t/dd00129', 74: '/m/014zdl', 75: '/m/07pxg6y', 76: '/m/07qsvvw', 77: '/m/04fq5q', 78: '/m/07rjzl8', 79: '/m/032s66', 80: '/m/0_1c', 81: '/m/07qz6j3', 82: '/m/07yv9', 83: '/m/0gvgw0', 84: '/m/02_41', 85: '/m/07p6mqd', 86: '/m/01jg02', 87: '/m/07p_0gm', 88: '/m/07pzfmf', 89: '/t/dd00067', 90: '/m/025_jnm', 91: '/m/01bjv', 92: '/m/0912c9', 93: '/m/04229', 94: '/m/07rkbfh', 95: '/m/0dls3', 96: '/m/073cg4', 97: '/m/07rv4dm', 98: '/m/04_sv', 99: '/m/07hvw1', 100: '/m/01hgjl', 101: '/m/01v1d8', 102: '/m/0cfdd', 103: '/m/0130jx', 104: '/m/02jz0l', 105: '/m/0838f', 106: '/m/01m2v', 107: '/m/06bz3', 108: '/m/03lty', 109: '/m/0dl5d', 110: '/m/02c8p', 111: '/m/07pjjrj', 112: '/t/dd00032', 113: '/m/07q2z82', 114: '/m/07rknqz', 115: '/m/0h9mv', 116: '/m/0ltv', 117: '/m/05rwpb', 118: '/m/04k94', 119: '/m/05zc1', 120: '/t/dd00077', 121: '/m/01p970', 122: '/m/03q5_w', 123: '/t/dd00127', 124: '/m/0j6m2', 125: '/m/01lsmm', 126: '/m/09ct_', 127: '/t/dd00092', 128: '/m/081rb', 129: '/m/07prgkl', 130: '/m/07s34ls', 131: '/m/0k4j', 132: '/t/dd00066', 133: '/m/0dv3j', 134: '/m/02_nn', 135: '/m/07rbp7_', 136: '/m/0z9c', 137: '/m/02mk9', 138: '/t/dd00130', 139: '/m/020bb7', 140: '/m/06_y0by', 141: '/m/0dl9sf8', 142: '/m/03vt0', 143: '/m/09xqv', 144: '/m/07cx4', 145: '/m/07pp8cl', 146: '/m/07qwyj0', 147: '/m/0c1dj', 148: '/m/068hy', 149: '/m/0bt9lr', 150: '/t/dd00136', 151: '/m/0brhx', 152: '/m/06by7', 153: '/m/07st88b', 154: '/m/09ld4', 155: '/m/07s04w4', 156: '/m/01b_21', 157: '/m/01z5f', 158: '/m/07pyy8b', 159: '/m/026z9', 160: '/m/0ggx5q', 161: '/m/07gxw', 162: '/m/025rv6n', 163: '/m/07st89h', 164: '/m/09b5t', 165: '/m/0mbct', 166: '/m/0xzly', 167: '/m/07qs1cx', 168: '/m/07pc8lb', 169: '/m/01z47d', 170: '/m/07q0yl5', 171: '/m/015p6', 172: '/m/07pggtn', 173: '/m/07r4wb8', 174: '/m/013y1f', 175: '/m/03xq_f', 176: '/m/03qjg', 177: '/m/085jw', 178: '/m/012ndj', 179: '/m/03j1ly', 180: '/m/03kmc9', 181: '/t/dd00018', 182: '/m/04zjc', 183: '/m/01xqw', 184: '/m/07y_7', 185: '/m/07pbtc8', 186: '/m/07rv9rh', 187: '/m/07rrh0c', 188: '/m/01jt3m', 189: '/m/0284vy3', 190: '/m/07rwm0c', 191: '/t/dd00004', 192: '/m/03m9d0z', 193: '/m/05kq4', 194: '/m/0hsrw', 195: '/m/01kcd', 196: '/m/06ncr', 197: '/m/07c6l', 198: '/m/03qtwd', 199: '/m/07qfr4h', 200: '/m/07pl1bw', 201: '/m/083vt', 202: '/m/01glhc', 203: '/t/dd00033', 204: '/m/02rr_', 205: '/m/07m2kt', 206: '/m/012f08', 207: '/m/02x984l', 208: '/m/096m7z', 209: '/m/07pczhz', 210: '/m/07rn7sz', 211: '/m/07s72n', 212: '/m/09t49', 213: '/m/01hsr_', 214: '/m/02cjck', 215: '/m/0llzx', 216: '/m/04s8yn', 217: '/m/07r5c2p', 218: '/m/0dwsp', 219: '/m/0dwtp', 220: '/m/0j45pbj', 221: '/m/02yds9', 222: '/m/01xq0k1', 223: '/m/07rpkh9', 224: '/m/06rqw', 225: '/m/014yck', 226: '/m/0cmf2', 227: '/m/0k5j', 228: '/m/01m4t', 229: '/m/0319l', 230: '/m/05pd6', 231: '/m/02mscn', 232: '/m/0l14qv', 233: '/m/05w3f', 234: '/m/01jg1z', 235: '/m/028v0c', 236: '/m/07qn4z3', 237: '/m/07rgkc5', 238: '/m/03r5q_', 239: '/m/07phhsh', 240: '/m/0145m', 241: '/m/0164x2', 242: '/m/07rgt08', 243: '/m/07qqyl4', 244: '/m/07r660_', 245: '/m/07sq110', 246: '/m/07swgks', 247: '/m/0155w', 248: '/m/06j6l', 249: '/m/03p19w', 250: '/m/07pkxdp', 251: '/m/04rmv', 252: '/m/01h3n', 253: '/m/0h2mp', 254: '/m/03v3yw', 255: '/m/07r4gkf', 256: '/m/07pws3f', 257: '/m/04cvmfc', 258: '/m/0dgbq', 259: '/m/02pjr4', 260: '/m/05lls', 261: '/m/0y4f8', 262: '/m/02bm9n', 263: '/m/0l7xg', 264: '/m/07qyrcz', 265: '/m/02bk07', 266: '/m/07ptzwd', 267: '/m/07bgp', 268: '/m/07q0h5t', 269: '/m/06xkwv', 270: '/m/07rcgpl', 271: '/t/dd00013', 272: '/m/07gql', 273: '/m/012n7d', 274: '/m/04qvtq', 275: '/t/dd00035', 276: '/m/0g6b5', 277: '/m/07r81j2', 278: '/m/07s8j8t', 279: '/m/07sbbz2', 280: '/m/06hck5', 281: '/m/06wzb', 282: '/m/026fgl', 283: '/m/0f8s22', 284: '/m/0261r1', 285: '/m/07mzm6', 286: '/m/074ft', 287: '/m/07q5rw0', 288: '/m/04rzd', 289: '/m/0ln16', 290: '/m/0l14jd', 291: '/m/05r6t', 292: '/m/02k_mr', 293: '/m/05x_td', 294: '/m/0fqfqc', 295: '/m/01z7dr', 296: '/m/0l14gg', 297: '/m/07qv4k0', 298: '/m/02g901', 299: '/m/09ddx', 300: '/m/0dbvp', 301: '/m/04gxbd', 302: '/m/07phxs1', 303: '/m/02p0sh1', 304: '/m/0mkg', 305: '/m/0316dw', 306: '/m/01j4z9', 307: '/m/0l14l2', 308: '/m/03k3r', 309: '/m/07qwdck', 310: '/m/0lyf6', 311: '/m/01lyv', 312: '/m/0gg8l', 313: '/m/0c3f7m', 314: '/t/dd00031', 315: '/m/0g12c5', 316: '/m/07pp_mv', 317: '/m/01d380', 318: '/m/07k1x', 319: '/m/0_ksk', 320: '/t/dd00005', 321: '/m/07qwf61', 322: '/m/0192l', 323: '/m/07r4k75', 324: '/m/06hps', 325: '/m/0l14j_', 326: '/m/02x8m', 327: '/m/0m0jc', 328: '/m/05r5wn', 329: '/m/04wptg', 330: '/m/07r04', 331: '/m/0199g', 332: '/m/015jpf', 333: '/m/07q6cd_', 334: '/m/04gy_2', 335: '/m/01yrx', 336: '/m/07qrkrw', 337: '/t/dd00034', 338: '/m/07qnq_y', 339: '/m/05tny_', 340: '/m/07qh7jl', 341: '/m/07pb8fc', 342: '/t/dd00065', 343: '/m/01hnzm', 344: '/m/07p6fty', 345: '/m/056ks2', 346: '/m/03wvsk', 347: '/m/07brj', 348: '/m/01rd7k', 349: '/m/07svc2k', 350: '/m/02v2lh', 351: '/m/0fd3y', 352: '/m/06w87', 353: '/m/023vsd', 354: '/m/07pdjhy', 355: '/m/07qlwh6', 356: '/t/dd00088', 357: '/m/03cczk', 358: '/m/07qjznl', 359: '/m/07r5v4s', 360: '/m/07p7b8y', 361: '/t/dd00037', 362: '/m/02z32qm', 363: '/m/0b9m1', 364: '/m/08j51y', 365: '/m/0chx_', 366: '/m/02p01q', 367: '/m/0btp2', 368: '/m/02mfyn', 369: '/m/07r_25d', 370: '/m/0h0rv', 371: '/m/01280g', 372: '/m/023pjk', 373: '/m/04brg2', 374: '/m/07s0s5r', 375: '/m/0dwt5', 376: '/m/02w4v', 377: '/m/01s0ps', 378: '/m/03gvt', 379: '/m/05148p4', 380: '/m/05r5c', 381: '/m/07q7njn', 382: '/m/068zj', 383: '/m/0939n_', 384: '/m/018j2', 385: '/m/034srq', 386: '/m/06q74', 387: '/m/07qfgpx', 388: '/m/0l156b', 389: '/m/07r67yg', 390: '/m/0140xf', 391: '/m/07qv_x_', 392: '/m/02l6bg', 393: '/m/01w250', 394: '/m/0gywn', 395: '/t/dd00002', 396: '/m/07s12q4', 397: '/m/02qmj0d', 398: '/m/03mb9', 399: '/m/039jq', 400: '/m/0ghcn6', 401: '/m/0l156k', 402: '/m/0g293', 403: '/m/01b82r', 404: '/m/01h8n0', 405: '/m/06h7j', 406: '/m/06_fw', 407: '/m/07pn_8q', 408: '/m/0395lw', 409: '/m/0642b4', 410: '/m/07s2xch', 411: '/m/01c194', 412: '/m/03w41f', 413: '/m/02rtxlg', 414: '/m/07qc9xj', 415: '/m/01x3z', 416: '/m/046dlr', 417: '/m/02bxd', 418: '/t/dd00048', 419: '/m/016cjb', 420: '/m/07s02z0', 421: '/m/03q5t', 422: '/m/07rdhzs', 423: '/m/07qdb04', 424: '/m/0242l', 425: '/m/05fw6t', 426: '/m/0dl83', 427: '/m/07qn5dc', 428: '/m/015vgc', 429: '/m/0ytgt', 430: '/m/02fxyj', 431: '/m/07qw_06', 432: '/m/0d31p', 433: '/t/dd00135', 434: '/m/06mb1', 435: '/m/0ngt1', 436: '/m/07bjf', 437: '/m/0dv5r', 438: '/t/dd00006', 439: '/m/01bns_', 440: '/t/dd00112', 441: '/m/01y3hg', 442: '/m/01h82_', 443: '/m/01swy6', 444: '/m/0cj0r', 445: '/m/0jb2l', 446: '/m/07pqc89', 447: '/m/06j64v', 448: '/m/0hdsk', 449: '/m/0l15bq', 450: '/m/01jwx6', 451: '/m/028sqc', 452: '/m/07p78v5', 453: '/m/027m70_', 454: '/m/016622', 455: '/m/08cyft', 456: '/m/01g90h', 457: '/m/07qjznt', 458: '/m/0cdnk', 459: '/m/05_wcq', 460: '/m/07c52', 461: '/m/02rhddq', 462: '/m/07pjwq1', 463: '/m/03m5k', 464: '/m/0ggq0m', 465: '/m/03l9g', 466: '/m/07s0dtb', 467: '/m/03cl9h', 468: '/m/07kc_', 469: '/m/0jtg0', 470: '/m/0k65p', 471: '/m/07r10fb', 472: '/m/01j3sz', 473: '/t/dd00001', 474: '/m/03wwcy', 475: '/m/012xff', 476: '/m/07qv_d5', 477: '/m/01d3sd', 478: '/m/07sx8x_', 479: '/m/0gy1t2s', 480: '/m/07qf0zm', 481: '/m/0l14t7', 482: '/m/01sm1g', 483: '/m/028ght', 484: '/m/09f96', 485: '/m/053hz1', 486: '/m/03_d0', 487: '/m/07r_k2n', 488: '/m/07rc7d9', 489: '/m/01v_m0', 490: '/m/0c2wf', 491: '/m/07pdhp0', 492: '/m/0glt670', 493: '/m/06bxc', 494: '/m/064t9', 495: '/m/0j2kx', 496: '/m/07pyf11', 497: '/m/07plct2', 498: '/m/0463cq4', 499: '/m/07p9k1k', 500: '/m/0dxrf', 501: '/m/024dl', 502: '/m/0150b9', 503: '/m/01wy6', 504: '/m/07pqn27', 505: '/m/0dq0md', 506: '/m/02p3nc', 507: '/m/07n_g', 508: '/m/025td0t', 509: '/m/07qcx4z', 510: '/m/0326g', 511: '/m/08p9q4', 512: '/m/018w8', 513: '/m/06cqb', 514: '/m/0790c', 515: '/m/05zppz', 516: '/m/0195fx', 517: '/m/03dnzn', 518: '/m/01b9nn', 519: '/m/07ptfmf', 520: '/m/02qldy', 521: '/t/dd00134', 522: '/m/02zsn', 523: '/t/dd00038', 524: '/m/05rj2', 525: '/m/032n05', 526: '/m/07ppn3j'}
beats_str2id = {v: k for (k,v) in beats_id2str.items()}

ced2beats_mapping = {}
with open("/star-data/xiaoyu/icefall_multi_KD/egs/librispeech/ASR/downloads/audioset/class_labels_indices.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, row in enumerate(reader):
        if i == 0:
            continue
        ced2beats_mapping[int(row[0])] = beats_str2id[row[1]]

def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


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
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model. Do not recommend to use this for AT",
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
        "--num-events", type=int, default=527, help="Number of sound events"
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
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
        default=30,
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
        "--use-KD",
        type=str2bool,
        default=True,
        help="If use KD target instead of gt label"
    )

    parser.add_argument(
        "--use-beats",
        type=str2bool,
        default=False,
        help="If use beats as teacher"
    )

    add_model_arguments(parser)

    return parser


def _str2modulelist(s: str, add_dot: bool = True):
    if add_dot:
        return [ss.strip() + "." for ss in s.split(",")] if s is not None else None
    else:
        return [ss.strip() for ss in s.split(",")] if s is not None else None


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
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


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


def get_model(params: AttributeDict) -> nn.Module:

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)

    model = AudioTaggingModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        num_events=params.num_events,
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

def extract_beats_embeddings(cuts):
    beats_embeddings = collate_custom_field(
        cuts, "beats_embedding", pad_value=-100
    ) # (N,C)
    return beats_embeddings

def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
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
        A batch of data. See `lhotse.dataset.AudioTaggingDataset()`
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

    supervisions = batch["supervisions"]
    cuts = supervisions["cut"]
    events = supervisions[
        "audio_event"
    ]  # the label indices are in CED format (https://github.com/RicherMans/CED)
    if params.use_beats:
        labels, _ = str2multihot(events, id_mapping=ced2beats_mapping)
    else:
        labels, _ = str2multihot(events, n_classes=params.num_events)

    # KD training target
    if params.use_KD and is_training:
        cuts_pre_mixed = [c if isinstance(c, MonoCut) else c.tracks[0].cut for c in cuts]
        labels = extract_beats_embeddings(cuts_pre_mixed)
    labels = labels.to(device)

    feature_lens = supervisions["num_frames"].to(device)

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    with torch.set_grad_enabled(is_training):
        loss = model(
            x=feature,
            x_lens=feature_lens,
            target=labels,
        )

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()

    return loss, info


def str2multihot(events: List[str], n_classes=527, id_mapping=None):
    # Convert strings separated by semi-colon to multi-hot class labels
    # input: ["0;1", "1;2"]
    # output: torch.tensor([[1,1,0], [0,1,1]])
    labels = [list(map(int, event.split(";"))) for event in events]
    batch_size = len(labels)
    out = torch.zeros(batch_size, n_classes)

    for i, label in enumerate(labels):
        if id_mapping is not None:
            label = [id_mapping[lb] for lb in label]
        out[i, label] = 1

    return out, labels


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
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
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
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

    num_samples = 0
    for batch_idx, batch in enumerate(train_dl):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        params.batch_idx_train += 1
        batch_size = batch["inputs"].size(0)
        num_samples += batch_size

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            save_bad_model()
            display_and_save_batch(batch, params=params)
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

        if batch_idx % 100 == 0 and params.use_fp16:
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
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
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
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

        if params.weighted_sampler and num_samples > params.num_samples:
            logging.info(f"Number of training samples exceeds {params.num_samples} in this epoch, move on to next epoch")
            break

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

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

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
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model,
            lr=params.base_lr,
            include_names=True,
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

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

    audioset = AudioSetATDatamodule(args)
    if params.use_beats:
        teacher = "beats"
    else:
        teacher = "CED"
    logging.info(f"Teacher model type: {teacher}")
    train_cuts = audioset.audioset_KD_train_cuts(teacher=teacher)

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 30.0:
            return False

        return True

    if not params.weighted_sampler:
        train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = audioset.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = audioset.audioset_eval_cuts()
    valid_dl = audioset.valid_dataloaders(valid_cuts)

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
            train_dl=train_dl,
            valid_dl=valid_dl,
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
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.AudioTaggingDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(
                batch,
                params=params,
            )
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    AudioSetATDatamodule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
