# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
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

from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface

from icefall.utils import add_sos, make_pad_mask, AttributeDict
from scaling import ScaledLinear

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling


class MultiKDModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        encoder_dim: int = 384,
        whisper_dim: int = 768,
        use_beats: bool = True,
        use_ecapa: bool = True,
        use_whisper: bool = True,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        
        # KD modules
        self.use_beats = use_beats
        if use_beats:
            self.beats_decoder = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(encoder_dim, 527),
            ) # 527 classes
        else:
            self.beats_decoder = None
        
        self.use_ecapa = use_ecapa
        if use_ecapa:
            self.ecapa_asp = AttentiveStatisticsPooling(channels=encoder_dim)
            self.ecapa_linear = nn.Linear(2 * encoder_dim, 192 ) # fixed 192-D vector
        else:
            self.ecapa_asp = None
            self.ecapa_linear = None
            
        self.use_whisper = use_whisper
        if use_whisper:
            self.whisper_projection = nn.Linear(encoder_dim, 2 * whisper_dim) # a linear transform
        else:
            self.whisper_projection = None

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        teacher_beats_embeddings: torch.Tensor = None,
        teacher_ecapa_embeddings: torch.Tensor = None,
        teacher_whisper_embeddings: torch.Tensor = None,
        teacher_whisper_embedding_lens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens) # (N,T,C)

        # beats loss
        if self.use_beats and teacher_beats_embeddings is not None:
            beats_logits = self.forward_beats(encoder_out, encoder_out_lens)
            
            # normalize the teacher probabilities
            teacher_beats_embeddings = teacher_beats_embeddings / teacher_beats_embeddings.sum(dim=-1).unsqueeze(-1).expand_as(teacher_beats_embeddings)
            
            beats_loss = F.kl_div(beats_logits, teacher_beats_embeddings, reduction="sum")
        else:
            beats_loss = None
        
        # ecapa loss
        if self.use_ecapa and teacher_ecapa_embeddings is not None:
            encoder_out = encoder_out.permute(0,2,1)
            ecapa_embeddings = self.ecapa_asp(encoder_out) # (N,C,T)
            encoder_out = encoder_out.permute(0,2,1)
            ecapa_embeddings = ecapa_embeddings.permute(0,2,1)
            ecapa_embeddings = self.ecapa_linear(ecapa_embeddings) # (N, 1, 192)
            ecapa_loss = 1 - F.cosine_similarity(ecapa_embeddings, teacher_ecapa_embeddings, dim=-1, eps=1e-6)
            ecapa_loss = ecapa_loss.sum()
        else:
            ecapa_loss = None
        
        if self.use_whisper and teacher_whisper_embeddings is not None:
            whisper_embeddings = self.forward_whisper(encoder_out, encoder_out_lens)
            teacher_whisper_embeddings = self.concat_successive_whisper_embeddings(
                whisper_embeddings,
                teacher_whisper_embeddings,
            )
            whisper_loss = F.l1_loss(whisper_embeddings, teacher_whisper_embeddings, reduction="none")
            mask = make_pad_mask(encoder_out_lens)
            whisper_loss.masked_fill_(mask.unsqueeze(-1), 0.0)
            # whisper_loss = whisper_loss.sum()/((~mask).sum() * teacher_whisper_embeddings.shape[-1])
            whisper_loss = whisper_loss.sum() / teacher_whisper_embeddings.shape[-1]
        else:
            whisper_loss = None
        
        return beats_loss, ecapa_loss, whisper_loss
    
    def forward_beats(
        self,
		encoder_out: torch.Tensor,
  		encoder_out_lens: torch.Tensor,
	):
        beats_logits = self.beats_decoder(encoder_out) # (N,)
        padding_mask = make_pad_mask(encoder_out_lens) 
        beats_logits[padding_mask] = 0
        beats_logits = beats_logits.sum(dim=1)
        beats_logits = beats_logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(beats_logits)
        beats_logits = torch.log_softmax(beats_logits, dim=-1).unsqueeze(1)
        
        return beats_logits
    
    def forward_ecapa(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        encoder_out = encoder_out.permute(0,2,1)
        encoder_out_lens = encoder_out_lens / torch.max(encoder_out_lens)
        ecapa_embeddings = self.ecapa_asp(encoder_out, encoder_out_lens) # (N,C,T)
        ecapa_embeddings = ecapa_embeddings.permute(0,2,1)
        ecapa_embeddings = self.ecapa_linear(ecapa_embeddings) # (N, 1, 192)
        
        return ecapa_embeddings
    
    def forward_whisper(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        return self.whisper_projection(F.relu(encoder_out))
    
    @staticmethod
    def concat_successive_whisper_embeddings(encoder_out, whisper_embeddings):
        t_expected = encoder_out.shape[1]
        N, T, C = whisper_embeddings.shape
        
        if T >= t_expected * 2:
            whisper_embeddings = whisper_embeddings[:, : t_expected * 2, :]
            
        whisper_embeddings = whisper_embeddings.reshape(N, t_expected, C * 2)
        assert whisper_embeddings.shape[1] == encoder_out.shape[1]
        return whisper_embeddings
    

class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        freeze_encoder: bool = False,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          freeze_encoder:
            Whether to freeze the parameters in encoder and encoder_embed
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder

        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )
            
        self.freeze_encoder = freeze_encoder

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=encoder_out_lens,
            target_lengths=target_lengths,
            reduction="sum",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        with torch.set_grad_enabled(not self.freeze_encoder):
            if self.freeze_encoder: # If freezing the encoder, set them to eval mode
                self.encoder.eval()
                self.encoder_embed.eval()
            encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(x.device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss



class SpeakerModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        encoder_dim: int = 384,
        num_spkrs: int = 2000,
        freeze_encoder: bool = False,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim

        self.ecapa_asp = AttentiveStatisticsPooling(channels=encoder_dim)
        self.ecapa_linear = nn.Linear(2 * encoder_dim, 192) # fixed 192-D vector
        self.classifier = nn.Linear(192, num_spkrs)
        
        self.freeze_encoder = freeze_encoder
        self.forward_ecapa = self.forward_speaker

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          target:
            The ground truth label of speaker IDs (N)
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        # Compute encoder outputs
        with torch.set_grad_enabled(not self.freeze_encoder):
            if self.freeze_encoder: # If freezing the encoder, set them to eval mode
                self.encoder.eval()
                self.encoder_embed.eval()
            encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
        
        # Forward the speaker module
        ecapa_embeddings = self.forward_speaker(encoder_out, encoder_out_lens)
        ecapa_embeddings = ecapa_embeddings.squeeze(1) # (N, 192)
        
        logits = self.classifier(ecapa_embeddings) # (N, 1, num_spkrs)
        loss = F.cross_entropy(logits, target, reduction="mean")
        
        return loss
    
    def forward_speaker(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        encoder_out = encoder_out.permute(0,2,1)
        encoder_out_lens = encoder_out_lens / torch.max(encoder_out_lens)
        ecapa_embeddings = self.ecapa_asp(encoder_out, encoder_out_lens) # (N,C,T)
        ecapa_embeddings = ecapa_embeddings.permute(0,2,1)
        ecapa_embeddings = self.ecapa_linear(ecapa_embeddings) # (N, 1, 192)
        
        return ecapa_embeddings
