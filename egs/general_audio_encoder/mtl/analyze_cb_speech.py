import torch
import numpy as np
from lhotse import load_manifest_lazy, CutSet

from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score

def load_codebook_indexes(c):
    info = c.codebook_indexes
    filename = info["path"]
    cb_index = np.load(filename)
    return cb_index

def analyze_codebook():
    N = 16
    input_manifest = "data/vq_hubert_large_layer_21_normalize_1_cb_16/librispeech_cuts_dev-clean.jsonl.gz"
    cuts = load_manifest_lazy(input_manifest)
    
    all_cb_index = []
    for cut in cuts:
        cb_index = load_codebook_indexes(cut)
        all_cb_index.append(torch.from_numpy(cb_index))
        
    all_cb_index = torch.cat(all_cb_index, dim=0) # (T, num_cb)
    
    for n in range(N):
        cb_index = all_cb_index[:, n]
        count = torch.bincount(cb_index, minlength=256)
        count = count / count.sum() # normalize to a distribution
        
        h = entropy(count.numpy())
        print(f"The entropy of the {n}-th codebook is {h}")
        
if __name__=="__main__":
    analyze_codebook()
    