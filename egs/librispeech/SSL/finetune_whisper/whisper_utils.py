import torch
import torch.nn.functional as F
import whisper
from whisper.audio import log_mel_spectrogram


def forward(self, x: torch.Tensor):
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    x = (x + self.positional_embedding[: x.shape[1], :]).to(x.dtype)

    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x


def replace_whisper_encoder_forward():
    """
    This function monkey patches the forward method of the whisper encoder.
    To be called before the model is loaded, it changes whisper to process audio with any length < 30s.
    """
    whisper.model.AudioEncoder.forward = forward


def compute_whisper_fbank(audio, n_mels=128):
    mel = log_mel_spectrogram(audio, n_mels=n_mels)
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)
    return mel