import random

import torch
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import TDTLossNumba

device = torch.device("cuda")

B, T, U, V = 4, 8, 4, 8  # here V is number of non blank labels
durations = [0, 1, 2, 3, 4, 5]
sigma = 0.05

acts = torch.rand([B, T, U, V + 1 + len(durations)]).to(device)
labels = [[random.randrange(0, V) for i in range(U - 1)] for j in range(B)]

import pdb; pdb.set_trace()
fn_pt = TDTLossNumba(blank=V, reduction='sum', durations=durations, sigma=sigma)
lengths = [acts.shape[1]] * acts.shape[0]
label_lengths = [len(l) for l in labels]
labels = torch.LongTensor(labels).to(device)
lengths = torch.LongTensor(lengths).to(device)
label_lengths = torch.LongTensor(label_lengths).to(device)



loss = fn_pt(acts, labels, lengths, label_lengths)
print(loss)