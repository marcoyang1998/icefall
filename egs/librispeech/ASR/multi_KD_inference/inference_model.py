import argparse

from model import MultiKDModel
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from zipformer import Zipformer2

from icefall.utils import str2bool

import kaldifeat
import torchaudio
import torch
import torch.nn as nn

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

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
        "--speaker-input-idx",
        type=int,
        default=-1,
        help="Which layer's output to be used for speaker embeddings. Start from 0."
    )

    parser.add_argument(
        "--whisper-dim",
        type=int,
        default=1280,
        help="The dimension of the whisper features",
        choices=[768, 1024, 1280],
    )

    parser.add_argument(
        "--use-subsampled_output",
        type=str2bool,
        default=True,
        help="If use the last subsampled output"
    )

    parser.add_argument(
        "--delta-t",
        type=int,
        default=0,
        help="The delta when computing whisper KD loss, only be used for causal model"
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--extract-spkr-embed",
        type=str2bool,
        default=False,
        help="If True, extracts the 192-D speaker embedding vector; otherwise return the"
        "encoder out features before speaker module (B,T,C)"
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="The path to the audio"
    )

    return parser

def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))

def get_encoder_embed(params) -> nn.Module:
    encoder_embed = Conv2dSubsampling(
        in_channels=80,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed

def get_encoder_model(params) -> nn.Module:
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

def get_model(params) -> nn.Module:
    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)

    model = MultiKDModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        use_beats=True,
        use_ecapa=True,
        use_whisper=True,
        whisper_dim=params.whisper_dim,
        speaker_input_idx=params.speaker_input_idx,
        mvq_KD=False,
        use_subsampled_output=params.use_subsampled_output,
        delta_t=params.delta_t if params.causal else 0,
    )

    return model

def main(args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # load model
    model = get_model(args)
    model.to(device)

    model.load_state_dict(
        torch.load(args.ckpt_path)["model"], strict=False
    )
    model.eval()

    # fbank extractor
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000 # only support 16k audio
    opts.mel_opts.num_bins = 80 # 80-bin
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    # load audio
    audio, fs = torchaudio.load(args.audio)
    assert fs == 16000
    audio_lens = audio.shape[1]
    audios = [audio.squeeze()]
    feature = fbank(audios)
    feature_lens = [f.size(0) for f in feature]

    feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True).to(device)
    feature_lens = torch.tensor(feature_lens, device=device)

    # batch inference
    encoder_out, encoder_out_lens, spkr_embedding = model.get_embeddings(
        feature,
        feature_lens,
        extract_spkr_embed=args.extract_spkr_embed,
    )
    print(encoder_out.shape)
    print(spkr_embedding.shape)

    at_logits = model.forward_beats(
        encoder_out, encoder_out_lens
    )
    top5 = at_logits.topk(5)
    print(f"The topk label are {top5}")

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
