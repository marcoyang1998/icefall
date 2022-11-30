# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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


import k2
import torch
import torch.nn as nn
import random
import time
from encoder_interface import EncoderInterface

from icefall.utils import add_sos
from scaling import penalize_abs_values_gt


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

        self.simple_am_proj = nn.Linear(
            encoder_dim, vocab_size,
        )
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        self_align: bool = True,
    ) -> torch.Tensor:
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
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

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
            (x.size(0), 4), dtype=torch.int64, device=x.device
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        #if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        #if self.training and random.random() < 0.25:
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
        
        if self_align:
            start = time.time()
            one_best_loss = self_alignment_loss3(
                logits=logits,
                ranges=ranges,
                y=y_padded,
                x_lens=x_lens
            )
            print(f"Time for calculating one best loss is: {time.time() - start}")
        else:
            one_best_loss = None
          
        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return (simple_loss, pruned_loss, one_best_loss)

def get_blank_connection(ranges: torch.Tensor, x_lens: torch.Tensor):                                                                                 
    # ranges: (B,T,S), S is the prune range, usually 5
    # x_lens: (B,) length of valid ranges
    # output a tensor blk_connections of same shape as ranges, (B,T,S) as the last t has no where to go                       
    # if blk_connections[t,s] < 0, means it cannot have a blank connection to t+1                                               
    # if blk_connections[t,s] = k, (k non-negative) means it can have a blank connection to k-th (0 indexed) state at t+1       
    bs = ranges.size(0)                                                                                                         
    T = ranges.size(1)                                                                                                          
    S = ranges.size(2)                                                                                                          
    blk_connections = []
    device = ranges.device                                                                          
                                                                                                                            
    for b in range(bs):                                                                                                         
        cur_range = ranges[b] # (T,S)                                                                                           
        lower_bound = ranges[b,:, 0] # the lower bound of all t                                                                 
        connections = torch.zeros(cur_range.shape).to(device).int()
        connections[:-1, :] = cur_range[:-1, :] - lower_bound[1:].unsqueeze(-1)                                                         
        connections[x_lens[b] -1:, :] = -1 # set all padding position to -1 to forbid blank transitions
        blk_connections.append(connections)                                                                               
                                                                                                                            
    return torch.stack(blk_connections)  
  
def build_logp_fsa(N, T, U, C, blank_connextions, x_lens) -> 'k2.Fsa':
    # now only test for N == 1. support N > 1 later.
    # assert N == 1
    assert U == 5
    # n = 0
    logp_vec = []
    stride_N = T * U * C
    for n in range(N):
      tot_states = T * U # T * 5
      arcs = ""
      for t in range(x_lens[n]):
        for s in range(U):
          cur_state_idx = t * U + s
          if (s < U - 1):
            for arc_idx in range(1, C):
                score_index = cur_state_idx * C + arc_idx + stride_N * n
                arcs += f"{cur_state_idx} {cur_state_idx + 1} {arc_idx} {score_index} {n} {t} {s} 0.0\n"

          next_state_idx1 = blank_connextions[n, t, s]
          if next_state_idx1 >= 0:
            score_index = cur_state_idx * C + stride_N * n
            next_state_idx01 = next_state_idx1 + (t + 1) * 5 
            arcs += f"{cur_state_idx} {next_state_idx01} {0} {score_index} {n} {t} {s} 0.0\n"
        

      # set super-final arc
      final_state = tot_states
      # for s in range(U):
      cur_state_idx = final_state - 1
      arcs += f"{cur_state_idx} {final_state} {-1} -1 {n} 0 0 0.0\n"


      arcs += f"{final_state}"
      # ans = k2.Fsa.from_str(arcs, aux_label_names=['aux_labels', 'score_index'])
      ans = k2.Fsa.from_str(arcs, aux_label_names=['score_index', 'n', 't', 'u'])

      # print(ans)
      ans = k2.arc_sort(ans)
      logp_vec.append(ans)
    return k2.create_fsa_vec(logp_vec)
  
  
def build_logp_fsa2(N, T, U, C, blank_connextions, ranges, y, x_lens) -> 'k2.Fsa':
    # now only test for N == 1. support N > 1 later.
    # assert N == 1
    assert U == 5
    # n = 0
    logp_vec = []
    stride_N = T * U * C
    for n in range(N):
      tot_states = T * U # T * 5
      arcs = ""
      for t in range(x_lens[n]):
        lower_bound = ranges[n,t,0]
        for s in range(U):
          cur_state_idx = t * U + s
          if (s < U - 1):
            #actual_u = s + lower_bound
            actual_u = ranges[n,t,s]
            arc_idx = y[n, actual_u]
            score_index = cur_state_idx * C + arc_idx + stride_N * n
            arcs += f"{cur_state_idx} {cur_state_idx + 1} {arc_idx} {score_index} {n} {t} {s} 0.0\n"
          # if (s < U - 1):
          #   for arc_idx in range(1, C):
          #       score_index = cur_state_idx * C + arc_idx + stride_N * n
          #       arcs += f"{cur_state_idx} {cur_state_idx + 1} {arc_idx} {score_index} {n} {t} {s} 0.0\n"

          next_state_idx1 = blank_connextions[n, t, s]
          if next_state_idx1 >= 0:
            score_index = cur_state_idx * C + stride_N * n
            next_state_idx01 = next_state_idx1 + (t + 1) * 5 
            arcs += f"{cur_state_idx} {next_state_idx01} {0} {score_index} {n} {t} {s} 0.0\n"
        

      # set super-final arc
      final_state = tot_states
      # for s in range(U):
      cur_state_idx = final_state - 1
      arcs += f"{cur_state_idx} {final_state} {-1} -1 {n} 0 0 0.0\n"


      arcs += f"{final_state}"
      # ans = k2.Fsa.from_str(arcs, aux_label_names=['aux_labels', 'score_index'])
      ans = k2.Fsa.from_str(arcs, aux_label_names=['score_index', 'n', 't', 'u'])

      # print(ans)
      ans = k2.arc_sort(ans)
      logp_vec.append(ans)
    return k2.create_fsa_vec(logp_vec)

def build_rnnt_topology(y: k2.RaggedTensor, bs: int):
    target_fsas = []
    
    for b in range(bs):
        linear_fsa = k2.add_epsilon_self_loops(k2.linear_fsa(y[b].tolist()))
        target_fsas.append(linear_fsa)
    
    return k2.create_fsa_vec(target_fsas)

def self_alignment_loss(logits: torch.Tensor, ranges: torch.Tensor, x_lens: torch.Tensor, y: k2.RaggedTensor):
    # logits: (B,T,S,V), pruned lattice
    # ranges: (B,T,S), valid range in the pruned lattice
    # y: groundtruth output tokens
    device = logits.device
    N,T,U,C = logits.size()
    blank_connextions = get_blank_connection(ranges, x_lens)
    logp = logits.log_softmax(dim=-1)
    logp_graph = build_logp_fsa(N, T, U, C, blank_connextions, x_lens)
    logp_graph = logp_graph.to(device)
    prefix_logp = torch.cat([torch.tensor([0]).to(device), logp.reshape(-1)])
    logp_graph.scores = torch.index_select(prefix_logp.to(torch.float32), dim=0, index=logp_graph.score_index.to(torch.int64) + 1)
    target_fsa = build_rnnt_topology(y, N)
    target_fsa = target_fsa.to(device)
    
    lattice = k2.intersect(target_fsa, logp_graph, treat_epsilons_specially=False)
    one_best = k2.shortest_path(lattice, use_double_scores=True)
    
    alignment = one_best.score_index
    labels = one_best.labels
    scores = one_best.scores
    
    t = one_best.t.long()
    u = one_best.u.long()
    n = one_best.n.long()
    nodes = logits[n,t,u]
    
    pseudo_emitted = nodes.argmax(-1)
    mask = pseudo_emitted == labels
    if random.random() > 0.9:
        print(f" {sum(mask).item()}/{len(mask)} = {sum(mask).item()/len(mask)} tokens are the same")
    
    only_non_blank = False
    if only_non_blank:
        non_blank_mask = labels > 0
        loss = -one_best.scores[non_blank_mask].sum()
    else:
        loss = -one_best.scores.sum()
    
    return loss
  
def self_alignment_loss2(logits: torch.Tensor, ranges: torch.Tensor, x_lens: torch.Tensor, y: k2.RaggedTensor):
    # logits: (B,T,S,V), pruned lattice
    # ranges: (B,T,S), valid range in the pruned lattice
    # y: groundtruth output tokens
    device = logits.device
    N,T,U,C = logits.size()
    blank_connextions = get_blank_connection(ranges, x_lens)
    logp = logits.log_softmax(dim=-1)
    logp_graph = build_logp_fsa2(N, T, U, C, blank_connextions, ranges, y, x_lens)
    logp_graph = logp_graph.to(device)
    prefix_logp = torch.cat([torch.tensor([0]).to(device), logp.reshape(-1)])
    logp_graph.scores = torch.index_select(prefix_logp.to(torch.float32), dim=0, index=logp_graph.score_index.to(torch.int64) + 1)
    target_fsa = build_rnnt_topology(y, N)
    target_fsa = target_fsa.to(device)
    
    lattice = k2.intersect(target_fsa, logp_graph, treat_epsilons_specially=False)
    one_best = k2.shortest_path(lattice, use_double_scores=True)
    
    alignment = one_best.score_index
    labels = one_best.labels
    scores = one_best.scores
    
    t = one_best.t.long()
    u = one_best.u.long()
    n = one_best.n.long()
    nodes = logits[n,t,u]
    
    pseudo_emitted = nodes.argmax(-1)
    mask = pseudo_emitted == labels
    if random.random() > 0.9:
        print(f" {sum(mask).item()}/{len(mask)} = {sum(mask).item()/len(mask)} tokens are the same")
    
    only_non_blank = False
    if only_non_blank:
        non_blank_mask = labels > 0
        loss = -one_best.scores[non_blank_mask].sum()
    else:
        loss = -one_best.scores.sum()
    
    return loss
  
def self_alignment_loss3(logits: torch.Tensor, ranges: torch.Tensor, x_lens: torch.Tensor, y: k2.RaggedTensor):
    # logits: (B,T,S,V), pruned lattice
    # ranges: (B,T,S), valid range in the pruned lattice
    # y: groundtruth output tokens
    device = logits.device
    N,T,U,C = logits.size()
    blank_connextions = get_blank_connection(ranges, x_lens)
    logp = logits.log_softmax(dim=-1)
    logp_graph = build_logp_fsa2(N, T, U, C, blank_connextions, ranges, y, x_lens)
    logp_graph = logp_graph.to(device)
    prefix_logp = torch.cat([torch.tensor([0]).to(device), logp.reshape(-1)])
    logp_graph.scores = torch.index_select(prefix_logp.to(torch.float32), dim=0, index=logp_graph.score_index.to(torch.int64) + 1)
    #target_fsa = build_rnnt_topology(y, N)
    #target_fsa = target_fsa.to(device)
    
    #lattice = k2.intersect(target_fsa, logp_graph, treat_epsilons_specially=False)
    one_best = k2.shortest_path(logp_graph, use_double_scores=True)
    
    alignment = one_best.score_index
    labels = one_best.labels
    scores = one_best.scores
    
    t = one_best.t.long()
    u = one_best.u.long()
    n = one_best.n.long()
    nodes = logits[n,t,u]
    
    pseudo_emitted = nodes.argmax(-1)
    mask = pseudo_emitted == labels
    if random.random() > 0.9:
        print(f" {sum(mask).item()}/{len(mask)} = {sum(mask).item()/len(mask)} tokens are the same")
    
    only_non_blank = False
    if only_non_blank:
        non_blank_mask = labels > 0
        loss = -one_best.scores[non_blank_mask].sum()
    else:
        loss = -one_best.scores.sum()
    
    return loss
  
if __name__=="__main__":
    ranges = torch.tensor([[0,1,2,3,4],                                                                                         
                           [0,1,2,3,4],                                                                                         
                           [1,2,3,4,5],                                                                                         
                           [3,4,5,6,7]]
                          )
    
    ranges = ranges.repeat(2,1,1)
    print(ranges)
    
    # valid length
    x_lens = torch.tensor([4,4])
    connections = get_blank_connection(ranges, x_lens)
    print(connections)
    
  