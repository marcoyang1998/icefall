import argparse
import csv
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from lhotse import load_manifest
from kd_datamodule import LibriSpeechKDDataModule

from train_multi_KD3 import add_model_arguments, get_model, get_params
from utils import get_class_dict, str2multihot

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

beats_id2str = {0: '/m/078jl', 1: '/m/07rjwbb', 2: '/m/04rlf', 3: '/m/07qb_dv', 4: '/t/dd00125', 5: '/m/02sgy', 6: '/m/0342h', 7: '/m/042v_gx', 8: '/m/04szw', 9: '/m/0b_fwt', 10: '/m/0fx80y', 11: '/m/07rqsjt', 12: '/m/018vs', 13: '/m/07szfh9', 14: '/t/dd00128', 15: '/m/02fsn', 16: '/m/0d8_n', 17: '/m/0l14_3', 18: '/m/07q4ntr', 19: '/m/07rwj3x', 20: '/m/09x0r', 21: '/m/01s0vc', 22: '/m/025wky1', 23: '/m/07ryjzk', 24: '/m/07xzm', 25: '/m/01qbl', 26: '/m/026t6', 27: '/m/0283d', 28: '/m/02hnl', 29: '/m/02lkt', 30: '/m/03qtq', 31: '/m/06rvn', 32: '/m/0bm02', 33: '/m/0l14md', 34: '/m/03qc9zr', 35: '/g/122z_qxw', 36: '/m/019jd', 37: '/m/02rlv9', 38: '/m/07qcpgn', 39: '/m/011k_j', 40: '/m/0239kh', 41: '/m/03t3fj', 42: '/m/07r_80w', 43: '/m/09d5_', 44: '/m/01g50p', 45: '/m/04zmvq', 46: '/m/06d_3', 47: '/m/07jdr', 48: '/m/07pt_g0', 49: '/m/015lz1', 50: '/m/07sr1lc', 51: '/t/dd00126', 52: '/m/07rrlb6', 53: '/m/0fx9l', 54: '/m/07qmpdm', 55: '/t/dd00121', 56: '/m/01yg9g', 57: '/m/07qlf79', 58: '/m/01jnbd', 59: '/m/03fwl', 60: '/m/0ch8v', 61: '/m/0jbk', 62: '/m/07plz5l', 63: '/t/dd00036', 64: '/t/dd00003', 65: '/m/07lnk', 66: '/m/02fs_r', 67: '/m/02cz_7', 68: '/m/02dgv', 69: '/m/02y_763', 70: '/m/015y_n', 71: '/m/04fgwm', 72: '/m/030rvx', 73: '/t/dd00129', 74: '/m/014zdl', 75: '/m/07pxg6y', 76: '/m/07qsvvw', 77: '/m/04fq5q', 78: '/m/07rjzl8', 79: '/m/032s66', 80: '/m/0_1c', 81: '/m/07qz6j3', 82: '/m/07yv9', 83: '/m/0gvgw0', 84: '/m/02_41', 85: '/m/07p6mqd', 86: '/m/01jg02', 87: '/m/07p_0gm', 88: '/m/07pzfmf', 89: '/t/dd00067', 90: '/m/025_jnm', 91: '/m/01bjv', 92: '/m/0912c9', 93: '/m/04229', 94: '/m/07rkbfh', 95: '/m/0dls3', 96: '/m/073cg4', 97: '/m/07rv4dm', 98: '/m/04_sv', 99: '/m/07hvw1', 100: '/m/01hgjl', 101: '/m/01v1d8', 102: '/m/0cfdd', 103: '/m/0130jx', 104: '/m/02jz0l', 105: '/m/0838f', 106: '/m/01m2v', 107: '/m/06bz3', 108: '/m/03lty', 109: '/m/0dl5d', 110: '/m/02c8p', 111: '/m/07pjjrj', 112: '/t/dd00032', 113: '/m/07q2z82', 114: '/m/07rknqz', 115: '/m/0h9mv', 116: '/m/0ltv', 117: '/m/05rwpb', 118: '/m/04k94', 119: '/m/05zc1', 120: '/t/dd00077', 121: '/m/01p970', 122: '/m/03q5_w', 123: '/t/dd00127', 124: '/m/0j6m2', 125: '/m/01lsmm', 126: '/m/09ct_', 127: '/t/dd00092', 128: '/m/081rb', 129: '/m/07prgkl', 130: '/m/07s34ls', 131: '/m/0k4j', 132: '/t/dd00066', 133: '/m/0dv3j', 134: '/m/02_nn', 135: '/m/07rbp7_', 136: '/m/0z9c', 137: '/m/02mk9', 138: '/t/dd00130', 139: '/m/020bb7', 140: '/m/06_y0by', 141: '/m/0dl9sf8', 142: '/m/03vt0', 143: '/m/09xqv', 144: '/m/07cx4', 145: '/m/07pp8cl', 146: '/m/07qwyj0', 147: '/m/0c1dj', 148: '/m/068hy', 149: '/m/0bt9lr', 150: '/t/dd00136', 151: '/m/0brhx', 152: '/m/06by7', 153: '/m/07st88b', 154: '/m/09ld4', 155: '/m/07s04w4', 156: '/m/01b_21', 157: '/m/01z5f', 158: '/m/07pyy8b', 159: '/m/026z9', 160: '/m/0ggx5q', 161: '/m/07gxw', 162: '/m/025rv6n', 163: '/m/07st89h', 164: '/m/09b5t', 165: '/m/0mbct', 166: '/m/0xzly', 167: '/m/07qs1cx', 168: '/m/07pc8lb', 169: '/m/01z47d', 170: '/m/07q0yl5', 171: '/m/015p6', 172: '/m/07pggtn', 173: '/m/07r4wb8', 174: '/m/013y1f', 175: '/m/03xq_f', 176: '/m/03qjg', 177: '/m/085jw', 178: '/m/012ndj', 179: '/m/03j1ly', 180: '/m/03kmc9', 181: '/t/dd00018', 182: '/m/04zjc', 183: '/m/01xqw', 184: '/m/07y_7', 185: '/m/07pbtc8', 186: '/m/07rv9rh', 187: '/m/07rrh0c', 188: '/m/01jt3m', 189: '/m/0284vy3', 190: '/m/07rwm0c', 191: '/t/dd00004', 192: '/m/03m9d0z', 193: '/m/05kq4', 194: '/m/0hsrw', 195: '/m/01kcd', 196: '/m/06ncr', 197: '/m/07c6l', 198: '/m/03qtwd', 199: '/m/07qfr4h', 200: '/m/07pl1bw', 201: '/m/083vt', 202: '/m/01glhc', 203: '/t/dd00033', 204: '/m/02rr_', 205: '/m/07m2kt', 206: '/m/012f08', 207: '/m/02x984l', 208: '/m/096m7z', 209: '/m/07pczhz', 210: '/m/07rn7sz', 211: '/m/07s72n', 212: '/m/09t49', 213: '/m/01hsr_', 214: '/m/02cjck', 215: '/m/0llzx', 216: '/m/04s8yn', 217: '/m/07r5c2p', 218: '/m/0dwsp', 219: '/m/0dwtp', 220: '/m/0j45pbj', 221: '/m/02yds9', 222: '/m/01xq0k1', 223: '/m/07rpkh9', 224: '/m/06rqw', 225: '/m/014yck', 226: '/m/0cmf2', 227: '/m/0k5j', 228: '/m/01m4t', 229: '/m/0319l', 230: '/m/05pd6', 231: '/m/02mscn', 232: '/m/0l14qv', 233: '/m/05w3f', 234: '/m/01jg1z', 235: '/m/028v0c', 236: '/m/07qn4z3', 237: '/m/07rgkc5', 238: '/m/03r5q_', 239: '/m/07phhsh', 240: '/m/0145m', 241: '/m/0164x2', 242: '/m/07rgt08', 243: '/m/07qqyl4', 244: '/m/07r660_', 245: '/m/07sq110', 246: '/m/07swgks', 247: '/m/0155w', 248: '/m/06j6l', 249: '/m/03p19w', 250: '/m/07pkxdp', 251: '/m/04rmv', 252: '/m/01h3n', 253: '/m/0h2mp', 254: '/m/03v3yw', 255: '/m/07r4gkf', 256: '/m/07pws3f', 257: '/m/04cvmfc', 258: '/m/0dgbq', 259: '/m/02pjr4', 260: '/m/05lls', 261: '/m/0y4f8', 262: '/m/02bm9n', 263: '/m/0l7xg', 264: '/m/07qyrcz', 265: '/m/02bk07', 266: '/m/07ptzwd', 267: '/m/07bgp', 268: '/m/07q0h5t', 269: '/m/06xkwv', 270: '/m/07rcgpl', 271: '/t/dd00013', 272: '/m/07gql', 273: '/m/012n7d', 274: '/m/04qvtq', 275: '/t/dd00035', 276: '/m/0g6b5', 277: '/m/07r81j2', 278: '/m/07s8j8t', 279: '/m/07sbbz2', 280: '/m/06hck5', 281: '/m/06wzb', 282: '/m/026fgl', 283: '/m/0f8s22', 284: '/m/0261r1', 285: '/m/07mzm6', 286: '/m/074ft', 287: '/m/07q5rw0', 288: '/m/04rzd', 289: '/m/0ln16', 290: '/m/0l14jd', 291: '/m/05r6t', 292: '/m/02k_mr', 293: '/m/05x_td', 294: '/m/0fqfqc', 295: '/m/01z7dr', 296: '/m/0l14gg', 297: '/m/07qv4k0', 298: '/m/02g901', 299: '/m/09ddx', 300: '/m/0dbvp', 301: '/m/04gxbd', 302: '/m/07phxs1', 303: '/m/02p0sh1', 304: '/m/0mkg', 305: '/m/0316dw', 306: '/m/01j4z9', 307: '/m/0l14l2', 308: '/m/03k3r', 309: '/m/07qwdck', 310: '/m/0lyf6', 311: '/m/01lyv', 312: '/m/0gg8l', 313: '/m/0c3f7m', 314: '/t/dd00031', 315: '/m/0g12c5', 316: '/m/07pp_mv', 317: '/m/01d380', 318: '/m/07k1x', 319: '/m/0_ksk', 320: '/t/dd00005', 321: '/m/07qwf61', 322: '/m/0192l', 323: '/m/07r4k75', 324: '/m/06hps', 325: '/m/0l14j_', 326: '/m/02x8m', 327: '/m/0m0jc', 328: '/m/05r5wn', 329: '/m/04wptg', 330: '/m/07r04', 331: '/m/0199g', 332: '/m/015jpf', 333: '/m/07q6cd_', 334: '/m/04gy_2', 335: '/m/01yrx', 336: '/m/07qrkrw', 337: '/t/dd00034', 338: '/m/07qnq_y', 339: '/m/05tny_', 340: '/m/07qh7jl', 341: '/m/07pb8fc', 342: '/t/dd00065', 343: '/m/01hnzm', 344: '/m/07p6fty', 345: '/m/056ks2', 346: '/m/03wvsk', 347: '/m/07brj', 348: '/m/01rd7k', 349: '/m/07svc2k', 350: '/m/02v2lh', 351: '/m/0fd3y', 352: '/m/06w87', 353: '/m/023vsd', 354: '/m/07pdjhy', 355: '/m/07qlwh6', 356: '/t/dd00088', 357: '/m/03cczk', 358: '/m/07qjznl', 359: '/m/07r5v4s', 360: '/m/07p7b8y', 361: '/t/dd00037', 362: '/m/02z32qm', 363: '/m/0b9m1', 364: '/m/08j51y', 365: '/m/0chx_', 366: '/m/02p01q', 367: '/m/0btp2', 368: '/m/02mfyn', 369: '/m/07r_25d', 370: '/m/0h0rv', 371: '/m/01280g', 372: '/m/023pjk', 373: '/m/04brg2', 374: '/m/07s0s5r', 375: '/m/0dwt5', 376: '/m/02w4v', 377: '/m/01s0ps', 378: '/m/03gvt', 379: '/m/05148p4', 380: '/m/05r5c', 381: '/m/07q7njn', 382: '/m/068zj', 383: '/m/0939n_', 384: '/m/018j2', 385: '/m/034srq', 386: '/m/06q74', 387: '/m/07qfgpx', 388: '/m/0l156b', 389: '/m/07r67yg', 390: '/m/0140xf', 391: '/m/07qv_x_', 392: '/m/02l6bg', 393: '/m/01w250', 394: '/m/0gywn', 395: '/t/dd00002', 396: '/m/07s12q4', 397: '/m/02qmj0d', 398: '/m/03mb9', 399: '/m/039jq', 400: '/m/0ghcn6', 401: '/m/0l156k', 402: '/m/0g293', 403: '/m/01b82r', 404: '/m/01h8n0', 405: '/m/06h7j', 406: '/m/06_fw', 407: '/m/07pn_8q', 408: '/m/0395lw', 409: '/m/0642b4', 410: '/m/07s2xch', 411: '/m/01c194', 412: '/m/03w41f', 413: '/m/02rtxlg', 414: '/m/07qc9xj', 415: '/m/01x3z', 416: '/m/046dlr', 417: '/m/02bxd', 418: '/t/dd00048', 419: '/m/016cjb', 420: '/m/07s02z0', 421: '/m/03q5t', 422: '/m/07rdhzs', 423: '/m/07qdb04', 424: '/m/0242l', 425: '/m/05fw6t', 426: '/m/0dl83', 427: '/m/07qn5dc', 428: '/m/015vgc', 429: '/m/0ytgt', 430: '/m/02fxyj', 431: '/m/07qw_06', 432: '/m/0d31p', 433: '/t/dd00135', 434: '/m/06mb1', 435: '/m/0ngt1', 436: '/m/07bjf', 437: '/m/0dv5r', 438: '/t/dd00006', 439: '/m/01bns_', 440: '/t/dd00112', 441: '/m/01y3hg', 442: '/m/01h82_', 443: '/m/01swy6', 444: '/m/0cj0r', 445: '/m/0jb2l', 446: '/m/07pqc89', 447: '/m/06j64v', 448: '/m/0hdsk', 449: '/m/0l15bq', 450: '/m/01jwx6', 451: '/m/028sqc', 452: '/m/07p78v5', 453: '/m/027m70_', 454: '/m/016622', 455: '/m/08cyft', 456: '/m/01g90h', 457: '/m/07qjznt', 458: '/m/0cdnk', 459: '/m/05_wcq', 460: '/m/07c52', 461: '/m/02rhddq', 462: '/m/07pjwq1', 463: '/m/03m5k', 464: '/m/0ggq0m', 465: '/m/03l9g', 466: '/m/07s0dtb', 467: '/m/03cl9h', 468: '/m/07kc_', 469: '/m/0jtg0', 470: '/m/0k65p', 471: '/m/07r10fb', 472: '/m/01j3sz', 473: '/t/dd00001', 474: '/m/03wwcy', 475: '/m/012xff', 476: '/m/07qv_d5', 477: '/m/01d3sd', 478: '/m/07sx8x_', 479: '/m/0gy1t2s', 480: '/m/07qf0zm', 481: '/m/0l14t7', 482: '/m/01sm1g', 483: '/m/028ght', 484: '/m/09f96', 485: '/m/053hz1', 486: '/m/03_d0', 487: '/m/07r_k2n', 488: '/m/07rc7d9', 489: '/m/01v_m0', 490: '/m/0c2wf', 491: '/m/07pdhp0', 492: '/m/0glt670', 493: '/m/06bxc', 494: '/m/064t9', 495: '/m/0j2kx', 496: '/m/07pyf11', 497: '/m/07plct2', 498: '/m/0463cq4', 499: '/m/07p9k1k', 500: '/m/0dxrf', 501: '/m/024dl', 502: '/m/0150b9', 503: '/m/01wy6', 504: '/m/07pqn27', 505: '/m/0dq0md', 506: '/m/02p3nc', 507: '/m/07n_g', 508: '/m/025td0t', 509: '/m/07qcx4z', 510: '/m/0326g', 511: '/m/08p9q4', 512: '/m/018w8', 513: '/m/06cqb', 514: '/m/0790c', 515: '/m/05zppz', 516: '/m/0195fx', 517: '/m/03dnzn', 518: '/m/01b9nn', 519: '/m/07ptfmf', 520: '/m/02qldy', 521: '/t/dd00134', 522: '/m/02zsn', 523: '/t/dd00038', 524: '/m/05rj2', 525: '/m/032n05', 526: '/m/07ppn3j'}
beats_str2id = {v: k for (k,v) in beats_id2str.items()}

ced2beats_mapping = {}
with open("downloads/audioset/class_labels_indices.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, row in enumerate(reader):
        if i == 0:
            continue
        ced2beats_mapping[int(row[0])] = beats_str2id[row[1]]

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
        "--trained-with-distillation",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--trained-with-multitask",
        type=str2bool,
        default=False,
    )
    
    parser.add_argument(
        "--freeze-encoder",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--num-events",
        type=int,
        default=527,
    )

    parser.add_argument(
        "--eval-subset",
        type=str,
        default="eval",
    )

    # parser.add_argument(
    #     "--decoder-dim",
    #     type=int,
    #     default=512,
    #     help="Embedding dimension in the decoder model.",
    # )

    # parser.add_argument(
    #     "--joiner-dim",
    #     type=int,
    #     default=512,
    #     help="""Dimension used in the joiner model.
    #     Outputs from the encoder and decoder model are projected
    #     to this dimension before adding.
    #     """,
    # )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--blank-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--do-audio-tagging",
        type=str2bool,
        default=True,
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
        "--freezing-encoder-layer-index",
        type=str,
        default="-1",
        help="Comma separated. start from 0, 0,1,2 means the first three encoder stacks are frozen",
    )

    parser.add_argument(
        "--freeze-encoder-steps",
        type=int,
        default=-1,
        help="Freeze the encoder for how many steps. Deactivated if `freeze-encoder` is True",
    )

    parser.add_argument(
        "--save-logits",
        type=str2bool,
        default=False,
    )

    add_model_arguments(parser)

    return parser

def inference_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
):
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    audio_event = supervisions["audio_event"]

    if params.trained_with_distillation:
        label, orig_labels = str2multihot(audio_event, id_mapping=ced2beats_mapping)
    else:
        label, orig_labels = str2multihot(audio_event)
    label = label.detach().cpu()

    feature_lens = supervisions["num_frames"].to(device)
    
    encoder_out, encoder_out_lens, _ = model.forward_encoder(feature, feature_lens, return_middle_out=True)
    
    # speaker
    audio_logits = model.forward_beats(encoder_out, encoder_out_lens)
    # sample_mean = audio_logits.mean(dim=-1)
    # audio_logits -= sample_mean.unsqueeze(-1)
    audio_logits = audio_logits.sigmoid().detach().cpu()
    
    return audio_logits, label
    

def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
) -> Dict:
    num_cuts = 0
    embedding_dict = {}
    teacher_embedding_dict = {}

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"
    
    all_logits = []
    all_labels = []
    saved_results = {
        "logits": {},
        "labels": {},
    }
        
    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        num_cuts += len(cut_ids)

        audio_logits, labels = inference_one_batch(
            params=params,
            model=model,
            batch=batch,
        )

        if params.save_logits:
            for id, logits, label in zip(cut_ids, audio_logits, labels):
                wav_name = id.split('/')[-1].replace(".wav", "")
                np.save(f"logits/logits_{wav_name}.npy", logits.detach().cpu().numpy())
                np.save(f"labels/labels_{wav_name}.npy", label.detach().cpu().numpy())
                saved_results["logits"][id] = f"logits/logits_{wav_name}.npy"
                saved_results["labels"][id] = f"labels/labels_{wav_name}.npy"
        
        all_logits.append(audio_logits)
        all_labels.append(labels)
        
        if batch_idx % 20 == 1:
            logging.info(f"Processed {num_cuts} cuts already.")
    logging.info(f"Finish collecting audio logits")
        
    return all_logits, all_labels, saved_results

@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechKDDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))
        
    params.res_dir = params.exp_dir / "inference_audio_tagging"
    
    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
        
    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"
        
    if params.causal:
        params.suffix += f"-chunk-size-{params.chunk_size}-left-context-frames-{params.left_context_frames}"
        
    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Evaluation started")
    
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        
    logging.info("About to create model")

    assert (params.trained_with_distillation and params.trained_with_multitask) == False

    if params.trained_with_distillation:
        from train_multi_KD3 import get_model
        model = get_model(params)
    elif params.trained_with_multitask:
        from train_asr import get_model
        model = get_model(params)
    else:
        from train_audio_tagging import get_model
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
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device), strict=False)
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device), strict=False)
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
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
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
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )

    model.to(device)
    model.eval()
    
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    args.return_cuts = True
    librispeech = LibriSpeechKDDataModule(args, device=device, evaluation=True)

    if params.eval_subset == "eval":
        audioset_cuts = librispeech.audioset_eval_cuts()
    elif params.eval_subset == "eval_all":
        audioset_cuts = librispeech.audioset_eval_all_cuts()
    else:
        raise NotImplementedError()

    audioset_dl = librispeech.valid_dataloaders(audioset_cuts)

    test_sets = [f"audioset_{params.eval_subset}"]

    logits, labels, saved_results = decode_dataset(
        dl=audioset_dl,
        params=params,
        model=model,
    )

    if params.save_logits:
        torch.save(saved_results, "zipformer_results.pt")

    logits = torch.cat(logits, dim=0).squeeze(dim=1).detach().numpy()
    labels = torch.cat(labels, dim=0).long().detach().numpy()

    mAP = average_precision_score(
        y_true=labels,
        y_score=logits,
    )

    logging.info(f"mAP for audioset {params.eval_subset} is: {mAP}")

    logging.info("Done")

if __name__ == "__main__":
    main()