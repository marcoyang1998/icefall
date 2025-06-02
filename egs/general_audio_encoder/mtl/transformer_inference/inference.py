import torchaudio
import torch
import torch.nn as nn

from audio_encoder import CausalAudioEncoder, AttributeDict

def get_420m_causal_audio_encoder() -> nn.Module:
    params = AttributeDict()
    params.causal = True
    params.encoder_dim= 1024
    params.num_layers = 24
    params.use_flash_attention = 1
    params.attention_dropout = 0.1
    params.num_heads = 16
    params.subsampling_factor = 4
    params.feature_dim = 128
    
    model = CausalAudioEncoder(
        config=params,
        num_mels=params.feature_dim, # fbank dim
    )
    return model

def _test_inference():
    model = get_420m_causal_audio_encoder()
    
    import pdb; pdb.set_trace()
    checkpoint_path = "transformer_finetune/exp-finetune-420m-causal-1-sub-4-ls-960--lr-3e-5-cosine-scheduler-warmup-12000-causal-1-freeze-encoder-0-freeze--1-step-encoder-lr-scale-1.0-from-hubert-large-mvq-cb16-delta-6-lh-large-giga-xl-pt-attn-drop-0.1-cosine-sched-with-musan-no-rir-400k/epoch-60.pt"
    state_dict = torch.load(checkpoint_path)["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    model.eval()
    device = torch.device("cuda")
    
    model.to(device)
    
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")
    
    from lhotse import load_manifest_lazy
    cuts = load_manifest_lazy("test.jsonl.gz")
    
    # we only support 16k hz
    audio = [
        torch.from_numpy(cut.load_audio()).to(device) for cut in cuts
    ]
    gt_fbank = torch.load("fbank_gt.pt")
    import pdb; pdb.set_trace()
    with torch.amp.autocast("cuda", enabled=True):
        fbank, fbank_len = model.compute_fbank(audio)
        assert torch.isclose(fbank, gt_fbank).all()
        encoder_out, encoder_out_lens = model(audio)
        encoder_out_gt = torch.load("encoder_out_gt.pt")
        encoder_out_len_gt = torch.load("encoder_out_len_gt.pt")
        assert torch.isclose(encoder_out_gt, encoder_out).all()
    
    import pdb; pdb.set_trace()
    print(encoder_out.shape)
    print(encoder_out_lens.shape)

def inference():
    # An example inference script
    model = get_420m_causal_audio_encoder()
    
    checkpoint_path = "transformer-420m-causal-mvq-pretrained-giga-xl-lh-large-cb16/iter-400000-avg-4.pt"
    state_dict = torch.load(checkpoint_path)["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    model.eval()
    device = torch.device("cuda")
    
    model.to(device)
    
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")
    
    # we only support 16k hz input audio
    audio = [
        torchaudio.load("transformer_inference/audio/84-121123-0024.flac")[0].to(device),
        torchaudio.load("transformer_inference/audio/84-121123-0028.flac")[0].to(device),
    ] # a list of pytorch tensor in shape [1, audio_len]
    with torch.amp.autocast("cuda", enabled=True):
        encoder_out, encoder_out_lens = model(audio)
    
    import pdb; pdb.set_trace()
    print(encoder_out.shape)
    print(encoder_out_lens)
    
    
if __name__=="__main__":
    # _test_inference()
    inference()