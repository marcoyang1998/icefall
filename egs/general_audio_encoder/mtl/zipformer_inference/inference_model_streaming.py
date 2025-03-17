import argparse
import math
from typing import Dict, List, Optional, Tuple

from model import MultiKDModel
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from zipformer import Zipformer2

from lhotse import Fbank, FbankConfig
import torchaudio
import torch
from torch import Tensor
import torch.nn as nn

LOG_EPS = math.log(1e-10)

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)

def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

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
        "--feature-dim",
        type=int,
        default=128,
        help="The dimension of fbank"
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
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
        in_channels=128,
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
        num_codebooks=0,
    )

    return model

def get_init_states(
    model: nn.Module,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """
    Returns a list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
    is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
    states[-2] is the cached left padding for ConvNeXt module,
    of shape (batch_size, num_channels, left_pad, num_freqs)
    states[-1] is processed_lens of shape (batch,), which records the number
    of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.
    """
    states = model.encoder.get_init_states(batch_size, device)

    embed_states = model.encoder_embed.get_init_states(batch_size, device)
    states.append(embed_states)

    processed_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    states.append(processed_lens)

    return states

def stack_states(state_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Stack list of zipformer states that correspond to separate utterances
    into a single emformer state, so that it can be used as an input for
    zipformer when those utterances are formed into a batch.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance. For element-n,
        state_list[n] is a list of cached tensors of all encoder layers. For layer-i,
        state_list[n][i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1,
        cached_val2, cached_conv1, cached_conv2).
        state_list[n][-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
        state_list[n][-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Note:
      It is the inverse of :func:`unstack_states`.
    """
    batch_size = len(state_list)
    assert (len(state_list[0]) - 2) % 6 == 0, len(state_list[0])
    tot_num_layers = (len(state_list[0]) - 2) // 6

    batch_states = []
    for layer in range(tot_num_layers):
        layer_offset = layer * 6
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key = torch.cat(
            [state_list[i][layer_offset] for i in range(batch_size)], dim=1
        )
        # cached_nonlin_attn: (num_heads, batch_size, left_context_len, head_dim)
        cached_nonlin_attn = torch.cat(
            [state_list[i][layer_offset + 1] for i in range(batch_size)], dim=1
        )
        # cached_val1: (left_context_len, batch_size, value_dim)
        cached_val1 = torch.cat(
            [state_list[i][layer_offset + 2] for i in range(batch_size)], dim=1
        )
        # cached_val2: (left_context_len, batch_size, value_dim)
        cached_val2 = torch.cat(
            [state_list[i][layer_offset + 3] for i in range(batch_size)], dim=1
        )
        # cached_conv1: (#batch, channels, left_pad)
        cached_conv1 = torch.cat(
            [state_list[i][layer_offset + 4] for i in range(batch_size)], dim=0
        )
        # cached_conv2: (#batch, channels, left_pad)
        cached_conv2 = torch.cat(
            [state_list[i][layer_offset + 5] for i in range(batch_size)], dim=0
        )
        batch_states += [
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
        ]

    cached_embed_left_pad = torch.cat(
        [state_list[i][-2] for i in range(batch_size)], dim=0
    )
    batch_states.append(cached_embed_left_pad)

    processed_lens = torch.cat([state_list[i][-1] for i in range(batch_size)], dim=0)
    batch_states.append(processed_lens)

    return batch_states

def unstack_states(batch_states: List[Tensor]) -> List[List[Tensor]]:
    """Unstack the zipformer state corresponding to a batch of utterances
    into a list of states, where the i-th entry is the state from the i-th
    utterance in the batch.

    Note:
      It is the inverse of :func:`stack_states`.

    Args:
        batch_states: A list of cached tensors of all encoder layers. For layer-i,
          states[i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1, cached_val2,
          cached_conv1, cached_conv2).
          state_list[-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
          states[-1] is processed_lens of shape (batch,), which records the number
          of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Returns:
        state_list: A list of list. Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance.
    """
    assert (len(batch_states) - 2) % 6 == 0, len(batch_states)
    tot_num_layers = (len(batch_states) - 2) // 6

    processed_lens = batch_states[-1]
    batch_size = processed_lens.shape[0]

    state_list = [[] for _ in range(batch_size)]

    for layer in range(tot_num_layers):
        layer_offset = layer * 6
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key_list = batch_states[layer_offset].chunk(chunks=batch_size, dim=1)
        # cached_nonlin_attn: (num_heads, batch_size, left_context_len, head_dim)
        cached_nonlin_attn_list = batch_states[layer_offset + 1].chunk(
            chunks=batch_size, dim=1
        )
        # cached_val1: (left_context_len, batch_size, value_dim)
        cached_val1_list = batch_states[layer_offset + 2].chunk(
            chunks=batch_size, dim=1
        )
        # cached_val2: (left_context_len, batch_size, value_dim)
        cached_val2_list = batch_states[layer_offset + 3].chunk(
            chunks=batch_size, dim=1
        )
        # cached_conv1: (#batch, channels, left_pad)
        cached_conv1_list = batch_states[layer_offset + 4].chunk(
            chunks=batch_size, dim=0
        )
        # cached_conv2: (#batch, channels, left_pad)
        cached_conv2_list = batch_states[layer_offset + 5].chunk(
            chunks=batch_size, dim=0
        )
        for i in range(batch_size):
            state_list[i] += [
                cached_key_list[i],
                cached_nonlin_attn_list[i],
                cached_val1_list[i],
                cached_val2_list[i],
                cached_conv1_list[i],
                cached_conv2_list[i],
            ]

    cached_embed_left_pad_list = batch_states[-2].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(cached_embed_left_pad_list[i])

    processed_lens_list = batch_states[-1].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(processed_lens_list[i])

    return state_list

def streaming_forward(
    features: Tensor,
    feature_lens: Tensor,
    model: nn.Module,
    states: List[Tensor],
    chunk_size: int,
    left_context_len: int,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """
    Returns encoder outputs, output lengths, and updated states.
    """
    cached_embed_left_pad = states[-2]
    (x, x_lens, new_cached_embed_left_pad,) = model.encoder_embed.streaming_forward(
        x=features,
        x_lens=feature_lens,
        cached_left_pad=cached_embed_left_pad,
    )
    assert x.size(1) == chunk_size, (x.size(1), chunk_size)

    src_key_padding_mask = make_pad_mask(x_lens)

    # processed_mask is used to mask out initial states
    processed_mask = torch.arange(left_context_len, device=x.device).expand(
        x.size(0), left_context_len
    )
    processed_lens = states[-1]  # (batch,)
    # (batch, left_context_size)
    processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
    # Update processed lengths
    new_processed_lens = processed_lens + x_lens

    # (batch, left_context_size + chunk_size)
    src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

    x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
    encoder_states = states[:-2]
    (
        encoder_out,
        encoder_out_lens,
        new_encoder_states,
    ) = model.encoder.streaming_forward(
        x=x,
        x_lens=x_lens,
        states=encoder_states,
        src_key_padding_mask=src_key_padding_mask,
    )
    encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

    new_states = new_encoder_states + [
        new_cached_embed_left_pad,
        new_processed_lens,
    ]
    return encoder_out, encoder_out_lens, new_states

def chunk_forward(
    audio: torch.Tensor,
    model: torch.nn.Module,
    feature_dim: int = 128,
    chunk_size: int = 8,
    left_context_frames: int = 256,
):
    # Perform chunk by chunk forward for the encoder. Each chunk is conditioned on the current chunk and left context (maintained by the states)
    # At each step, we take a chunk of audio and forward the encoder
    # For the first chunk, we wait until the accumulated audio duration to reacg (buffer + chunk_duration), the buffer
    # is necessary for the convolution subsampling module in the encoder.
    # After the first chunk, we process normal chunk-by-chunk inference when the accumulated audio reaches chunk_duration
    # An example of Buffer=2 frames, chunk=5 frames, the latency for the first chunk is 7 frames (as we need to accumulate 7 frames 
    # for decoding), the rest chunks have latency of 5 frames.
    # Each time we feed (5 + 2) frames to the encoder, and then shift 5 frames
    # Chunk 1: AAAAAAA
    # Chunk 2:      AAAAAAA
    # Chunk 3:           AAAAAAA
    
    # NOTE: params.chunk is the chunk_size regarding to the input of the zipformer encoder, so at fbank level, the chunk size
    # is 2 * params.chunk_size
    
    # fbank extractor
    extractor = Fbank(FbankConfig(num_mel_bins=feature_dim))
    
    device = next(model.parameters()).device
    
    chunk_size = int(chunk_size)
    chunk_size_samples = int(chunk_size * 2 * 160) # chunk size represented in audio samples of 16kHz sampling rate
    left_context_len = int(left_context_frames)
    pad_length = 7 + 2 * 3 # buffer required by encoder_embed module 
    pad_length_samples = (7 + 2 * 3) * 160 
    
    # intialize states, to be maintained during chunk-wise forward
    initial_states = get_init_states(model=model, batch_size=1, device=device)
    
    # start forward chunk by chunk
    encoder_outs = []
    encoder_out_lens = 0
    states = initial_states
    
    num_chunk = 0
    num_processed_samples = 0 # audio samples
    
    # the actual loop performing the chunk-wise inference of the encoder
    while True:
        # prepare the input for processing current chunk
        # compute fbank for the current chunk
        audio_chunk = audio[:, num_processed_samples: num_processed_samples + (chunk_size_samples + pad_length_samples)]
        features = extractor.extract(audio_chunk, sampling_rate=16000)
        features = features.to(device)
        feature_lens = features.shape[0]
        
        feature_lens = torch.tensor([feature_lens], device=device) # shape: (1)
        features = features.unsqueeze(0) # shape: (1,T,num_mels)
        
        # the audio chunk could be shorted than the expected length, for example in the last two chunks
        # pad the chunk so that the input shape is (chunk_size + buffer)
        tail_length = chunk_size * 2 + 7 + 2 * 3 # each prepared chunk should have this length
        if features.size(1) < tail_length:
            pad_length = tail_length - features.size(1)
            feature_lens += pad_length
            features = torch.nn.functional.pad(
                features,
                (0, 0, 0, pad_length),
                mode="constant",
                value=LOG_EPS,
            )
        
        states = stack_states([states])
    
        # forward current chunk in batch=1
        encoder_out, encoder_out_len, new_states = streaming_forward(
            features=features,
            feature_lens=feature_lens,
            model=model,
            states=states,
            chunk_size=chunk_size,
            left_context_len=left_context_len,
        )
        
        encoder_outs.append(encoder_out)
        encoder_out_lens += encoder_out_len
        
        # update the states
        states = unstack_states(new_states)[0]
        
        num_chunk += 1
        num_processed_samples += chunk_size_samples
        
        if num_processed_samples > audio.shape[1]:
            print(f"Audio is exhausted.")
            break
    
    import pdb; pdb.set_trace()
    encoder_outs = torch.cat(encoder_outs, dim=1) # shape: (1,T,C)
    
    return encoder_outs, encoder_out_lens
    


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

    # load audio
    import pdb; pdb.set_trace()
    audio, fs = torchaudio.load(args.audio)
    assert fs == 16000
    
    encoder_out, encoder_out_lens = chunk_forward(
        audio=audio, # shape (1, num_samples)
        model=model,
        feature_dim=args.feature_dim,
        chunk_size=args.chunk_size,
        left_context_frames=args.left_context_frames,
    )
    
    print(encoder_out.shape)

    at_logits = model.forward_audio_tagging(
        encoder_out, encoder_out_lens, return_logits=True
    )
    top5 = at_logits.topk(5)
    print(f"The topk label are {top5}")

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)