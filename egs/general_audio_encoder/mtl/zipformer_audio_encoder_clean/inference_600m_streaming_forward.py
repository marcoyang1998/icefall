import argparse
import math

from model import MultiKDModel
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from zipformer import Zipformer2

from lhotse import Fbank, FbankConfig
import torchaudio
import torch
import torch.nn as nn

LOG_EPS = math.log(1e-10)

class ZipformerConfig:
    def __init__(self):
        # 用 _config 存储所有参数
        self._config = {
            "feature_dim": 128,
            "pos_dim": 48,
            "output_downsampling_factor": 2,
            "downsampling_factor": "1,2,4,8,4,2",
            "num_encoder_layers": "2,2,3,4,3,2",
            "feedforward_dim": "512,768,1024,1536,1024,768",
            "encoder_dim": "192,256,448,768,448,192",
            "encoder_unmasked_dim": "192,192,256,256,256,192",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "num_heads": "4,4,4,8,4,4",
            "causal": True,
        }

    def __getattr__(self, key):
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"'ZipformerConfig' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "_config":
            super().__setattr__(key, value)
        else:
            self._config[key] = value

    def __delattr__(self, key):
        if key in self._config:
            del self._config[key]
        else:
            raise AttributeError(f"'ZipformerConfig' object has no attribute '{key}'")

    def to_dict(self):
        return dict(self._config)

    def __repr__(self):
        return f"ZipformerConfig({self._config})"
        


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
        "--model-version",
        type=str,
        default="600m_uniform_out_ds1",
    )
    
    parser.add_argument(
        "--causal",
        type=str2bool,
        default=True,
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
        output_downsampling_factor=params.output_downsampling_factor,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple("32"),
        pos_head_dim=_to_int_tuple("4"),
        value_head_dim=_to_int_tuple("12"),
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

def get_params(args):
    params = ZipformerConfig()
    params.chunk_size = args.chunk_size
    params.left_context_frames = args.left_context_frames
    
    model_version = args.model_version
    if model_version == "600m_uniform_out_ds1":
        params.output_downsampling_factor = 1
        params.downsampling_factor = "1,2,4,8,4,2,1"
        params.num_encoder_layers = "1,2,3,4,1,1,1"
        params.feedforward_dim = "3840,3840,3840,3840,3840,3840,3840"
        params.encoder_dim = "1280,1280,1280,1280,1280,1280,1280"
        params.encoder_unmasked_dim = "768,768,768,768,768,768,768"
        params.cnn_module_kernel = "31,31,15,15,15,31,31"
        params.num_heads = "8,8,8,8,8,8,8"
    elif model_version == "600m_uniform_out_ds2":
        params.output_downsampling_factor = 2
        params.downsampling_factor = "1,2,4,8,4,2,1"
        params.num_encoder_layers = "1,2,3,4,1,1,1"
        params.feedforward_dim = "3840,3840,3840,3840,3840,3840,3840"
        params.encoder_dim = "1280,1280,1280,1280,1280,1280,1280"
        params.encoder_unmasked_dim = "768,768,768,768,768,768,768"
        params.cnn_module_kernel = "31,31,15,15,15,31,31"
        params.num_heads = "8,8,8,8,8,8,8"
    else:
        raise ValueError()
    return params    
    

def get_model(model_version) -> nn.Module:
    # initialise the encoder model
    
    params = get_params(model_version)
    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)
    print(params)

    model = MultiKDModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        num_codebooks=0,
    )

    return model

def main(args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # load model
    model = get_model(args)
    model.to(device)

    info = model.load_state_dict(
        torch.load(args.ckpt_path)["model"], strict=False
    )
    print(info)
    model.eval()

    # fbank extractor
    extractor = Fbank(FbankConfig(num_mel_bins=128))

    # load audio
    audio, fs = torchaudio.load(args.audio)
    assert fs == 16000
    audio_lens = audio.shape[1]
    audios = audio.squeeze()
    feature = [extractor.extract(audios, sampling_rate=fs)]
    feature_lens = [f.size(0) for f in feature]

    feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True, padding_value=LOG_EPS).to(device)
    feature_lens = torch.tensor(feature_lens, device=device)
    print(f"Input fbank feature: {feature.shape}")

    # batch inference
    encoder_out, encoder_out_lens = model.forward_encoder(
        feature,
        feature_lens,
    )
    print(encoder_out.shape)
    torch.save(encoder_out, "encoder_out.pt")

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)