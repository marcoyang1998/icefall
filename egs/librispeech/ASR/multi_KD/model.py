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

from typing import Optional, Tuple, List
import logging
import random

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface

from icefall.utils import add_sos, make_pad_mask, AttributeDict
from scaling import ScaledLinear

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from multi_quantization.prediction import JointCodebookLoss


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
        speaker_input_idx: int = -1,
        mvq_KD: bool = False,
        num_codebooks: int = 32,
        mvq_kd_layer: int = -1,
        cb_input_dim: int = 512,
        use_subsampled_output: bool = True
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
          use_whisper:
            If use l1 embedding loss for whisper features
          mvq_KD:
            If use codebook loss for whisper teacher
          mvq_kd_layer:
            Use which zipformer block's output for MVQ-KD. If -1, use `encoder_out`, 
            Otherwise use middle_out[mvq_kd_layer][-1]
          speaker_input_idx:
            Which modules results to be inputted to the ecapa module
          use_subsampled_output:
            If use the final subsampled output i.e 25 Hz
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.encoder_dims = self.encoder.encoder_dim # tuple
        self.num_layers = self.encoder.num_encoder_layers # tuple
        
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
        self.speaker_input_idx = speaker_input_idx
        if use_ecapa:
            if speaker_input_idx == -1: # use the last layer
                self.ecapa_asp = AttentiveStatisticsPooling(channels=encoder_dim)
                self.ecapa_linear = nn.Linear(2 * encoder_dim, 192 ) # fixed 192-D vector
            else:
                speaker_input_dim = self.encoder_dims[speaker_input_idx]
                self.ecapa_asp = AttentiveStatisticsPooling(channels=speaker_input_dim)
                self.ecapa_linear = nn.Linear(2 * speaker_input_dim, 192 ) # fixed 192-D vector
        else:
            self.ecapa_asp = None
            self.ecapa_linear = None
            
        self.use_whisper = use_whisper
        if use_whisper:
            if use_subsampled_output:
                self.whisper_projection = nn.Linear(encoder_dim, 2 * whisper_dim) # a linear transform
            else:
                self.whisper_projection = nn.Linear(256, whisper_dim) # a linear transform
            self.codebook_loss_net = None
        else:
            self.whisper_projection = None

        # if use codebook loss
        self.mvq_KD = mvq_KD
        self.mvq_layer = mvq_kd_layer
        if mvq_KD:
            assert num_codebooks > 0
            self.codebook_loss_net = JointCodebookLoss(
                predictor_channels=cb_input_dim,
                num_codebooks=num_codebooks,
                is_joint=False,
            )
        else:
            self.codebook_loss_net = None


    def forward_encoder(
        self, 
        x: torch.Tensor,
        x_lens: torch.Tensor,
        return_middle_out: bool = False,
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

        encoder_out, encoder_out_lens, middle_out = self.encoder(x, x_lens, src_key_padding_mask, return_middle_out=return_middle_out)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens, middle_out

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor = None,
        teacher_beats_embeddings: torch.Tensor = None,
        teacher_ecapa_embeddings: torch.Tensor = None,
        teacher_whisper_embeddings: torch.Tensor = None,
        teacher_whisper_embedding_lens: torch.Tensor = None,
        teacher_whisper_codebook_indexes: torch.Tensor = None,
        teacher_whisper_codebook_indexes_lens: torch.Tensor = None,
        return_middle_out: bool = False,
        reduction: str = "sum"
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
          teacher_beats_embeddings:
            BEATs teacher's embedding (N, 1, 527)
          teacher_ecapa_embeddings:
            Ecapa teacher's embedding (N, 1, 192)
          teacher_whisper_embeddings:
            Whisper teachers embeddings (N,T,C)
          teacher_whisper_embedding_lens:
            The length of the whisper teacher's embedding (N,)
          teacher_whisper_codebook_indexes:
            The whisper teacher's embeddings encoded in codebook indexes (N,T, num_cb)
          teacher_whisper_codebook_indexes_lens:
            The length of the whisper teacher's codebook indexes (N,)

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
        # assert y.num_axes == 2, y.num_axes

        # assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens, middle_out = self.forward_encoder(x, x_lens, return_middle_out=True) # (N,T,C)

        # beats loss
        if self.use_beats and teacher_beats_embeddings is not None:
            beats_logits = self.forward_beats(encoder_out, encoder_out_lens)
            teacher_beats_embeddings = teacher_beats_embeddings.squeeze(dim=1) # (N, num_events)
            beats_loss = F.binary_cross_entropy_with_logits(beats_logits, teacher_beats_embeddings, reduction=reduction)
        else:
            beats_loss = None
        
        # ecapa loss
        if self.use_ecapa and teacher_ecapa_embeddings is not None:
            assert middle_out is not None
            if self.speaker_input_idx == -1:
                ecapa_embeddings = self.forward_ecapa(
                    encoder_out, # (N,T,C)
                    encoder_out_lens,
                )
            else:
                ecapa_input_embeddings = middle_out[self.speaker_input_idx] # a list of (T,N,C)
                # ecapa_input_embeddings = torch.mean(torch.stack(ecapa_input_embeddings), dim=0)
                ecapa_input_embeddings = sum(ecapa_input_embeddings) / len(ecapa_input_embeddings)
                ecapa_input_embeddings = ecapa_input_embeddings.permute(1,0,2)

                ecapa_embeddings = self.forward_ecapa(
                    ecapa_input_embeddings,
                    encoder_out_lens,
                )
            
            # encoder_out = encoder_out.permute(0,2,1)
            # ecapa_embeddings = self.ecapa_asp(encoder_out) # (N,C,T)
            # encoder_out = encoder_out.permute(0,2,1)
            # ecapa_embeddings = ecapa_embeddings.permute(0,2,1)
            # ecapa_embeddings = self.ecapa_linear(ecapa_embeddings) # (N, 1, 192)
            ecapa_loss = 1 - F.cosine_similarity(ecapa_embeddings, teacher_ecapa_embeddings, dim=-1, eps=1e-6)
            if reduction == "sum":
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
            if reduction == "sum":
                whisper_loss = whisper_loss.sum() / teacher_whisper_embeddings.shape[-1]
            else:
                whisper_loss = whisper_loss / teacher_whisper_embeddings.shape[-1]
        else:
            whisper_loss = None

        if (self.mvq_KD and self.training) and teacher_whisper_codebook_indexes is not None:
            if self.mvq_layer == -1:
                middle_layer = encoder_out
            else:
                # middle_layer = self.encoder.encoders[self.mvq_layer].upsample(middle_out[self.mvq_layer][-1]).permute(1,0,2)
                middle_layer = middle_out[self.mvq_layer][-1].permute(1,0,2)
            whisper_cb_loss = self.forward_codebook(
                middle_layer_output=middle_layer,
                codebook_indexes=teacher_whisper_codebook_indexes,
            )
        else:
            whisper_cb_loss = None
        
        return beats_loss, ecapa_loss, whisper_loss, whisper_cb_loss
    
    def forward_beats(
        self,
        encoder_out: torch.Tensor,
  		encoder_out_lens: torch.Tensor,
	):
        beats_logits = self.beats_decoder(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens) 
        beats_logits[padding_mask] = 0
        beats_logits = beats_logits.sum(dim=1)
        beats_logits = beats_logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(beats_logits) # (N, num_events)
        # beats_logits = torch.log_softmax(beats_logits, dim=-1).unsqueeze(1)
        
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

    def forward_codebook(
        self,
        middle_layer_output: torch.Tensor,
        codebook_indexes: torch.Tensor,
    ):
        """Calculate the codebook loss for the model (knowledge distillation)

        Args:
                middle_layer_output (torch.Tensor):
                        The embeddings extracted from the middle layer of the zipformer encoder
                codebook_indexes (torch.Tensor):
                        The encoded codebook indexes for knowledge distillation

        Returns:
                The codebook loss value
        """
        len_CI = codebook_indexes.size(1)
        len_mid_layer = middle_layer_output.size(1)
        ratio = round(len_CI / len_mid_layer)

        if ratio == 1:  # Having the same frame rate
            assert len_CI > len_mid_layer, (len_CI, len_mid_layer)
            codebook_indexes = codebook_indexes[:, :len_mid_layer, :]
            assert codebook_indexes.size(1) == middle_layer_output.size(1)
            codebook_loss = self.codebook_loss_net(
                middle_layer_output, codebook_indexes
            )
        elif ratio >= 2:
            codebook_indexes = self.concat_successive_codebook_indexes(
                middle_layer_output, codebook_indexes, ratio=ratio
            )
            codebook_loss = self.codebook_loss_net(
                middle_layer_output, codebook_indexes
            )

        return codebook_loss
    
    @staticmethod
    def concat_successive_whisper_embeddings(encoder_out, whisper_embeddings):
        t_expected = encoder_out.shape[1]
        N, T, C = whisper_embeddings.shape
        
        if T >= t_expected * 2:
            whisper_embeddings = whisper_embeddings[:, : t_expected * 2, :]
            whisper_embeddings = whisper_embeddings.reshape(N, t_expected, C * 2)
        else:
            whisper_embeddings = whisper_embeddings[:, : t_expected, :]

        # An ugly way to avoid shape mismatch of dummy embedding, this only happens if the whole batch is dummy embedding!
        if whisper_embeddings.shape[1] != encoder_out.shape[1]:
            return encoder_out

        assert whisper_embeddings.shape[1] == encoder_out.shape[1]
        assert whisper_embeddings.shape[2] == encoder_out.shape[2], f"{N}x{T}x{C} {whisper_embeddings.shape}, {encoder_out.shape}"

        return whisper_embeddings

    @staticmethod
    def concat_successive_codebook_indexes(middle_layer_output, codebook_indexes, ratio=2):
        # Output rate of hubert is 50 frames per second,
        # while that of current encoder is 25.
        # Following code handling two issues:
        # 1.
        #   Roughly speaking, to generate another frame output,
        #   hubert needes extra two frames,
        #   while current encoder needs extra four frames.
        #   Suppose there are only extra three frames provided,
        #   hubert will generate another frame while current encoder does nothing.
        # 2.
        #   codebook loss is a frame-wise loss, to enalbe 25 frames studnet output
        #   learns from 50 frames teacher output, two successive frames of teacher model
        #   output is concatenated together.
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape
        assert T >= t_expected, (T, t_expected)
        # Handling issue 1.
        if T >= t_expected * 2:
            codebook_indexes = codebook_indexes[:, : t_expected * ratio, :]
        if (
            T / t_expected < 1.1
        ):  # To be changed, dirty hack to jump out of this function
            codebook_indexes = codebook_indexes[:, :t_expected, :]
            assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
            return codebook_indexes
        # Handling issue 2.
        codebook_indexes = codebook_indexes.reshape(N, t_expected, C * ratio)
        assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
        return codebook_indexes
    


class MultiKDASRModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        whisper_dim: int = 768,
        use_beats: bool = True,
        use_ecapa: bool = True,
        use_whisper: bool = True,
        speaker_input_idx: int = -1,
        mvq_KD: bool = False,
        num_codebooks: int = 32,
        mvq_kd_layer: int = -1,
        cb_input_dim: int = 512,
        use_subsampled_output: bool = True,
        vocab_size: int = 500,
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
          use_whisper:
            If use l1 embedding loss for whisper features
          mvq_KD:
            If use codebook loss for whisper teacher
          mvq_kd_layer:
            Use which zipformer block's output for MVQ-KD. If -1, use `encoder_out`, 
            Otherwise use middle_out[mvq_kd_layer][-1]
          speaker_input_idx:
            Which modules results to be inputted to the ecapa module
          use_subsampled_output:
            If use the final subsampled output i.e 25 Hz
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.encoder_dims = self.encoder.encoder_dim # tuple
        self.num_layers = self.encoder.num_encoder_layers # tuple
        
        # Transducer modules
        self.use_transducer = True
        self.decoder = decoder
        self.joiner = joiner

        self.simple_am_proj = ScaledLinear(
            encoder_dim, vocab_size, initial_scale=0.25
        )
        self.simple_lm_proj = ScaledLinear(
            decoder_dim, vocab_size, initial_scale=0.25
        )
        
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
        self.speaker_input_idx = speaker_input_idx
        if use_ecapa:
            if speaker_input_idx == -1: # use the last layer
                self.ecapa_asp = AttentiveStatisticsPooling(channels=encoder_dim)
                self.ecapa_linear = nn.Linear(2 * encoder_dim, 192 ) # fixed 192-D vector
            else:
                speaker_input_dim = self.encoder_dims[speaker_input_idx]
                self.ecapa_asp = AttentiveStatisticsPooling(channels=speaker_input_dim)
                self.ecapa_linear = nn.Linear(2 * speaker_input_dim, 192 ) # fixed 192-D vector
        else:
            self.ecapa_asp = None
            self.ecapa_linear = None
            
        self.use_whisper = use_whisper
        if use_whisper:
            if use_subsampled_output:
                self.whisper_projection = nn.Linear(encoder_dim, 2 * whisper_dim) # a linear transform
            else:
                self.whisper_projection = nn.Linear(256, whisper_dim) # a linear transform
            self.codebook_loss_net = None
        else:
            self.whisper_projection = None

        # if use codebook loss
        self.mvq_KD = mvq_KD
        self.mvq_layer = mvq_kd_layer
        if mvq_KD:
            assert num_codebooks > 0
            self.codebook_loss_net = JointCodebookLoss(
                predictor_channels=cb_input_dim,
                num_codebooks=num_codebooks,
                is_joint=False,
            )
        else:
            self.codebook_loss_net = None


    def forward_encoder(
        self, 
        x: torch.Tensor,
        x_lens: torch.Tensor,
        return_middle_out: bool = False,
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

        encoder_out, encoder_out_lens, middle_out = self.encoder(x, x_lens, src_key_padding_mask, return_middle_out=return_middle_out)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens, middle_out

    
    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        reduction: str = "sum",
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

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction=reduction,
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
                reduction=reduction,
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
        teacher_beats_embeddings: torch.Tensor = None,
        teacher_ecapa_embeddings: torch.Tensor = None,
        teacher_whisper_embeddings: torch.Tensor = None,
        teacher_whisper_embedding_lens: torch.Tensor = None,
        teacher_whisper_codebook_indexes: torch.Tensor = None,
        teacher_whisper_codebook_indexes_lens: torch.Tensor = None,
        return_middle_out: bool = False,
        reduction: str = "sum"
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
          teacher_beats_embeddings:
            BEATs teacher's embedding (N, 1, 527)
          teacher_ecapa_embeddings:
            Ecapa teacher's embedding (N, 1, 192)
          teacher_whisper_embeddings:
            Whisper teachers embeddings (N,T,C)
          teacher_whisper_embedding_lens:
            The length of the whisper teacher's embedding (N,)
          teacher_whisper_codebook_indexes:
            The whisper teacher's embeddings encoded in codebook indexes (N,T, num_cb)
          teacher_whisper_codebook_indexes_lens:
            The length of the whisper teacher's codebook indexes (N,)

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
        encoder_out, encoder_out_lens, middle_out = self.forward_encoder(x, x_lens, return_middle_out=True) # (N,T,C)

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
                reduction=reduction,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        # beats loss
        if self.use_beats and teacher_beats_embeddings is not None:
            beats_logits = self.forward_beats(encoder_out, encoder_out_lens)
            
            # normalize the teacher probabilities
            # teacher_beats_embeddings = teacher_beats_embeddings / teacher_beats_embeddings.sum(dim=-1).unsqueeze(-1).expand_as(teacher_beats_embeddings)
            teacher_beats_embeddings = teacher_beats_embeddings.squeeze(dim=1) # (N, num_events)
            
            beats_loss = F.binary_cross_entropy_with_logits(beats_logits, teacher_beats_embeddings, reduction=reduction)
            # beats_loss = F.kl_div(beats_logits, teacher_beats_embeddings, reduction="sum")
        else:
            beats_loss = None
        
        # ecapa loss
        if self.use_ecapa and teacher_ecapa_embeddings is not None:
            assert middle_out is not None
            if self.speaker_input_idx == -1:
                ecapa_embeddings = self.forward_ecapa(
                    encoder_out, # (N,T,C)
                    encoder_out_lens,
                )
            else:
                ecapa_input_embeddings = middle_out[self.speaker_input_idx] # a list of (T,N,C)
                # ecapa_input_embeddings = torch.mean(torch.stack(ecapa_input_embeddings), dim=0)
                ecapa_input_embeddings = sum(ecapa_input_embeddings) / len(ecapa_input_embeddings)
                ecapa_input_embeddings = ecapa_input_embeddings.permute(1,0,2)

                ecapa_embeddings = self.forward_ecapa(
                    ecapa_input_embeddings,
                    encoder_out_lens,
                )
            
            # encoder_out = encoder_out.permute(0,2,1)
            # ecapa_embeddings = self.ecapa_asp(encoder_out) # (N,C,T)
            # encoder_out = encoder_out.permute(0,2,1)
            # ecapa_embeddings = ecapa_embeddings.permute(0,2,1)
            # ecapa_embeddings = self.ecapa_linear(ecapa_embeddings) # (N, 1, 192)
            ecapa_loss = 1 - F.cosine_similarity(ecapa_embeddings, teacher_ecapa_embeddings, dim=-1, eps=1e-6)
            if reduction == "sum":
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
            if reduction == "sum":
                whisper_loss = whisper_loss.sum() / teacher_whisper_embeddings.shape[-1]
            else:
                whisper_loss = whisper_loss / teacher_whisper_embeddings.shape[-1]
        else:
            whisper_loss = None

        if (self.mvq_KD and self.training) and teacher_whisper_codebook_indexes is not None:
            if self.mvq_layer == -1:
                middle_layer = encoder_out
            else:
                # middle_layer = self.encoder.encoders[self.mvq_layer].upsample(middle_out[self.mvq_layer][-1]).permute(1,0,2)
                middle_layer = middle_out[self.mvq_layer][-1].permute(1,0,2)
            whisper_cb_loss = self.forward_codebook(
                middle_layer_output=middle_layer,
                codebook_indexes=teacher_whisper_codebook_indexes,
            )
        else:
            whisper_cb_loss = None
        
        return beats_loss, ecapa_loss, whisper_loss, whisper_cb_loss, simple_loss, pruned_loss
    
    def forward_beats(
        self,
        encoder_out: torch.Tensor,
  		encoder_out_lens: torch.Tensor,
	):
        beats_logits = self.beats_decoder(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens) 
        beats_logits[padding_mask] = 0
        beats_logits = beats_logits.sum(dim=1)
        beats_logits = beats_logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(beats_logits) # (N, num_events)
        # beats_logits = torch.log_softmax(beats_logits, dim=-1).unsqueeze(1)
        
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

    def forward_codebook(
        self,
        middle_layer_output: torch.Tensor,
        codebook_indexes: torch.Tensor,
    ):
        """Calculate the codebook loss for the model (knowledge distillation)

        Args:
                middle_layer_output (torch.Tensor):
                        The embeddings extracted from the middle layer of the zipformer encoder
                codebook_indexes (torch.Tensor):
                        The encoded codebook indexes for knowledge distillation

        Returns:
                The codebook loss value
        """
        len_CI = codebook_indexes.size(1)
        len_mid_layer = middle_layer_output.size(1)
        ratio = round(len_CI / len_mid_layer)

        if ratio == 1:  # Having the same frame rate
            assert len_CI > len_mid_layer, (len_CI, len_mid_layer)
            codebook_indexes = codebook_indexes[:, :len_mid_layer, :]
            assert codebook_indexes.size(1) == middle_layer_output.size(1)
            codebook_loss = self.codebook_loss_net(
                middle_layer_output, codebook_indexes
            )
        elif ratio >= 2:
            codebook_indexes = self.concat_successive_codebook_indexes(
                middle_layer_output, codebook_indexes, ratio=ratio
            )
            codebook_loss = self.codebook_loss_net(
                middle_layer_output, codebook_indexes
            )

        return codebook_loss
    
    @staticmethod
    def concat_successive_whisper_embeddings(encoder_out, whisper_embeddings):
        t_expected = encoder_out.shape[1]
        N, T, C = whisper_embeddings.shape
        
        if T >= t_expected * 2:
            whisper_embeddings = whisper_embeddings[:, : t_expected * 2, :]
            whisper_embeddings = whisper_embeddings.reshape(N, t_expected, C * 2)
        else:
            whisper_embeddings = whisper_embeddings[:, : t_expected, :]

        # An ugly way to avoid shape mismatch of dummy embedding, this only happens if the whole batch is dummy embedding!
        if whisper_embeddings.shape[1] != encoder_out.shape[1]:
            return encoder_out

        # if whisper_embeddings.shape[2] != encoder_out.shape[2]:
        #     return encoder_out

        assert whisper_embeddings.shape[1] == encoder_out.shape[1]
        assert whisper_embeddings.shape[2] == encoder_out.shape[2]

        return whisper_embeddings

    @staticmethod
    def concat_successive_codebook_indexes(middle_layer_output, codebook_indexes, ratio=2):
        # Output rate of hubert is 50 frames per second,
        # while that of current encoder is 25.
        # Following code handling two issues:
        # 1.
        #   Roughly speaking, to generate another frame output,
        #   hubert needes extra two frames,
        #   while current encoder needs extra four frames.
        #   Suppose there are only extra three frames provided,
        #   hubert will generate another frame while current encoder does nothing.
        # 2.
        #   codebook loss is a frame-wise loss, to enalbe 25 frames studnet output
        #   learns from 50 frames teacher output, two successive frames of teacher model
        #   output is concatenated together.
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape
        assert T >= t_expected, (T, t_expected)
        # Handling issue 1.
        if T >= t_expected * 2:
            codebook_indexes = codebook_indexes[:, : t_expected * ratio, :]
        if (
            T / t_expected < 1.1
        ):  # To be changed, dirty hack to jump out of this function
            codebook_indexes = codebook_indexes[:, :t_expected, :]
            assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
            return codebook_indexes
        # Handling issue 2.
        codebook_indexes = codebook_indexes.reshape(N, t_expected, C * ratio)
        assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
        return codebook_indexes
   

class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_projection: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        audio_tagging_input_dim: int = 512,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        do_audio_tagging: bool = False,
        do_speaker_verification: bool = False,
        num_spkrs: int = -1,
        speaker_input_idx: int = -1,
        freeze_encoder: bool = False,
        freezing_encoder_layer_index: List[int] = [],
        freeze_encoder_steps: int = -1,
        sync_other_tasks: bool = False,
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
            Whether to freeze the parameters in encoder and encoder_embed. 
          freezing_encoder_layer_index:
            Specify which encoder layer should be frozen; Used when the speaker module
            use the middle out as input. Will be ignored in `freeze_encoder=True`
          freeze_encoder_steps:
            Specify for how many steps should the encoder be frozen. Will be ignored in `freeze_encoder=True`
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
        self.encoder_dim = encoder_dim
        self.encoder_dims = self.encoder.encoder_dim # tuple
        self.num_layers = self.encoder.num_encoder_layers
        
        
        self.whisper_projection = encoder_projection # can be None
        self.encoder_projection = self.whisper_projection

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

        self.do_audio_tagging = do_audio_tagging
        if do_audio_tagging:
            self.audio_tagging_decoder = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(audio_tagging_input_dim, 527),
            ) # 527 classes
            self.beats_decoder = self.audio_tagging_decoder
        
        self.do_speaker_verification = do_speaker_verification
        self.speaker_input_idx = speaker_input_idx
        if do_speaker_verification:
            if speaker_input_idx == -1: # use the last layer
                self.ecapa_asp = AttentiveStatisticsPooling(channels=encoder_dim)
                self.ecapa_linear = nn.Linear(2 * encoder_dim, 192 ) # fixed 192-D vector
            else:
                speaker_input_dim = self.encoder_dims[speaker_input_idx]
                self.ecapa_asp = AttentiveStatisticsPooling(channels=speaker_input_dim)
                self.ecapa_linear = nn.Linear(2 * speaker_input_dim, 192 ) # fixed 192-D vector
                
            self.speaker_classifier = nn.Linear(192, num_spkrs)
        else:
            self.ecapa_asp = None
            self.ecapa_linear = None
            self.speaker_classifier = None
        
        self.freeze_encoder = freeze_encoder
        self.freezing_encoder_layer_index = freezing_encoder_layer_index
        self.freeze_encoder_steps = freeze_encoder_steps

        self.sync_other_tasks = sync_other_tasks # if sync SV and AT tasks when freezing encoder

        self.forward_beats = self.forward_audio_tagging

    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        return_middle_out: bool=False,
        freezing_encoder_layer_index: List[int] = []
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

        # encoder_out, encoder_out_lens, middle_out = self.encoder(x, x_lens, src_key_padding_mask, return_middle_out=return_middle_out)
        outputs = self.encoder(
            x, 
            x_lens,
            src_key_padding_mask,
            return_middle_out=return_middle_out,
            freezing_layer_idx=freezing_encoder_layer_index,
        )
        if len(outputs) == 3:
            encoder_out, encoder_out_lens, middle_out = outputs
        else:
            encoder_out, encoder_out_lens = outputs
            middle_out = None

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens, middle_out

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
        reduction: str = "sum",
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

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction=reduction,
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
                reduction=reduction,
            )

        return simple_loss, pruned_loss

    def forward_audio_tagging(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        logits = self.beats_decoder(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens) # (N,T)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits) # (N, num_events)

        return logits
    
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
        

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        audio_tagging_label: torch.Tensor = None,
        speaker_label: torch.Tensor = None,
        return_middle_out: bool = False,
        batch_idx: int = 1e10 + 1,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
          audio_tagging_label:
            The multi-hot audio tagging label
          return_middle_out:
            Return the layer-wise output of encoder
          batch_idx:
            The absolute batch index
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

        _freeze_encoder = self.freeze_encoder or (batch_idx < self.freeze_encoder_steps)
        if batch_idx % 50 == 0 and self.training:
            logging.info(f"Freeze_encoder: {_freeze_encoder}; Current batch idx: {batch_idx}")

        # Compute encoder outputs
        with torch.set_grad_enabled(not _freeze_encoder):
            if _freeze_encoder: # If freezing the encoder, set them to eval mode
                self.encoder.eval()
                self.encoder_embed.eval()
            pre_projection_encoder_out, encoder_out_lens, middle_out = self.forward_encoder(
                x,
                x_lens,
                return_middle_out=True,
                freezing_encoder_layer_index=self.freezing_encoder_layer_index
            )

            if self.encoder_projection is not None:
                encoder_out = self.encoder_projection(pre_projection_encoder_out)
            else:
                encoder_out = pre_projection_encoder_out

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
                reduction=reduction,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        ctc_loss = torch.empty(0)

        if self.do_audio_tagging:
            _freeze_AT_module = _freeze_encoder and self.sync_other_tasks
            with torch.set_grad_enabled(not _freeze_AT_module):
                if _freeze_encoder:
                    self.beats_decoder.eval()
                audio_tagging_logits = self.forward_audio_tagging(
                    encoder_out=pre_projection_encoder_out,
                    encoder_out_lens=encoder_out_lens,
                )
            audio_tagging_loss = F.binary_cross_entropy_with_logits(
                audio_tagging_logits,
                audio_tagging_label,
                reduction=reduction
            )
        else:
            audio_tagging_loss = torch.empty(0)
            
        # ecapa loss
        if self.do_speaker_verification:
            _freeze_speaker_module = _freeze_encoder and self.sync_other_tasks
            with torch.set_grad_enabled(not _freeze_speaker_module):
                if _freeze_encoder:
                    self.ecapa_asp.eval()
                    self.ecapa_linear.eval()
                if self.speaker_input_idx == -1:
                    ecapa_embeddings = self.forward_speaker(
                        encoder_out, # (N,T,C)
                        encoder_out_lens,
                    )
                else:
                    assert middle_out is not None
                    ecapa_input_embeddings = middle_out[self.speaker_input_idx] # a list of (T,N,C)
                    ecapa_input_embeddings = sum(ecapa_input_embeddings) / len(ecapa_input_embeddings)
                    ecapa_input_embeddings = ecapa_input_embeddings.permute(1,0,2)

                    ecapa_embeddings = self.forward_speaker(
                        ecapa_input_embeddings,
                        encoder_out_lens,
                    ) # (N,1,192)
                
            ecapa_embeddings = ecapa_embeddings.squeeze(1) # (N, 192)
                
            logits = self.speaker_classifier(ecapa_embeddings) # (N, num_spkrs)
            sv_loss = F.cross_entropy(logits, speaker_label, reduction=reduction)
        else:
            sv_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss, audio_tagging_loss, sv_loss


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



class AudioTaggingModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        encoder_dim: int = 384,
        num_events: int = 527,
        freeze_encoder: bool = False,
    ):
        """An audio tagging model

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

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        ) # 527 classes
        
        # for multi-class classification
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.freeze_encoder = freeze_encoder

        self.forward_beats = self.forward_audio_tagging

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, return_middle_out: bool=True,
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

        encoder_out, encoder_out_lens, _ = self.encoder(x, x_lens, src_key_padding_mask, return_middle_out)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens, _

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
            The ground truth label of audio events, could be many hot
        Returns:
          Return the binary crossentropy loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        # Compute encoder outputs
        with torch.set_grad_enabled(not self.freeze_encoder):
            if self.freeze_encoder: # If freezing the encoder, set them to eval mode
                self.encoder.eval()
                self.encoder_embed.eval()
            encoder_out, encoder_out_lens, _ = self.forward_encoder(x, x_lens, return_middle_out=True)
        
        # Forward the speaker module
        logits = self.classifier(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)

        loss = self.criterion(logits, target)

        return loss

    def forward_audio_tagging(self, encoder_out, encoder_out_lens):
        logits = self.classifier(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)

        return logits



class GenreClassificationModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        encoder_dim: int = 384,
        num_genres: int = 10,
        freeze_encoder: bool = False,
    ):
        """An audio tagging model

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

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_genres),
        ) # 10 classes
        
        # for multi-class classification
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.freeze_encoder = freeze_encoder

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, return_middle_out: bool=True,
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

        encoder_out, encoder_out_lens, _ = self.encoder(x, x_lens, src_key_padding_mask, return_middle_out)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens, _

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
            The ground truth label of audio events, could be many hot
        Returns:
          Return the binary crossentropy loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        # Compute encoder outputs
        with torch.set_grad_enabled(not self.freeze_encoder):
            if self.freeze_encoder: # If freezing the encoder, set them to eval mode
                self.encoder.eval()
                self.encoder_embed.eval()
            encoder_out, encoder_out_lens, _ = self.forward_encoder(x, x_lens, return_middle_out=True)
        
        # Forward the classifier
        logits = self.forward_genre_classification(encoder_out, encoder_out_lens)
        loss = self.criterion(logits, target)

        return loss

    def forward_genre_classification(self, encoder_out, encoder_out_lens):
        logits = self.classifier(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)

        return logits

