import csv

import torch
import numpy as np
from lhotse import load_manifest_lazy, CutSet
import multi_quantization as quantization

from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score

def load_codebook_indexes(c):
    info = c.codebook_indexes
    if isinstance(info, dict):
        filename = info["path"]
        cb_index = np.load(filename)
    else:
        cb_index = c.load_custom("codebook_indexes")
    return cb_index        

def read_csv(filename):
    results = {}
    with open(filename, "r") as f:
        data = csv.reader(f, delimiter=',')
        
        for i, line in enumerate(data):
            if i == 0:
                continue
            audio, fold, target, category = line[:4]
            cut_name = audio.replace(".wav", "")
            results[cut_name] = target
    
    return results

def fix_manifest():
    csv_file = "download/ESC-50-master/meta/esc50.csv"
    meta_data = read_csv(csv_file)
    
    input_manifest = "data/fbank_esc/esc_cuts.jsonl.gz"
    cuts = load_manifest_lazy(input_manifest)

    new_cuts = []
    import pdb; pdb.set_trace()
    for cut in cuts:
        sound_event = meta_data[cut.id]
        cut.sound_event = sound_event
        new_cuts.append(cut)
        
    import pdb; pdb.set_trace()
    new_cuts = CutSet.from_cuts(new_cuts)
    output_manifest_name = "data/esc/esc_cuts.jsonl.gz"
    print(f"Saving the manfiest to {output_manifest_name}")
    new_cuts.to_jsonl(output_manifest_name)

def compute_codebook_mutual_information():
    N = 4
    
    input_manifest = f"data/vq_dasheng_large_layer_-1_normalize_0_cb_{N}/esc_cuts.jsonl.gz"
    cuts = load_manifest_lazy(input_manifest)
    
    with open(f"codebook_analysis_dasheng_cb{N}.txt", "w") as fout:
        for event in range(50):
            def filter_event(cut):
                if int(cut.sound_event) == event:
                    return True
                return False
            subset = cuts.filter(filter_event)
            
            all_cb_index = []
            for cut in subset:
                cb_index = cut.load_custom("codebook_indexes")
                all_cb_index.append(torch.from_numpy(cb_index))
            all_cb_index = torch.cat(all_cb_index, dim=0) # (T, num_cb)
            for n in range(N):
                cb_index = all_cb_index[:, n]
                count = torch.bincount(cb_index, minlength=256)
                values, indices = count.topk(5)
                
                print(f"For class: {event}, the {n}-th codebook: top 5 entries are: {indices}, their values are {values}", file=fout)
        
    labels = []
    all_cb_index = []
    for cut in cuts:
        label = cut.sound_event
        labels = labels + [label for _ in range(5 * 25)]
        
        cb_index = cut.load_custom("codebook_indexes")
        all_cb_index.append(torch.from_numpy(cb_index))
        
    all_cb_index = torch.cat(all_cb_index, dim=0) # (T, num_cb)
    
    for i in range(N):
        mi = mutual_info_score(all_cb_index[:, i], labels)
        print(f"Mutual information between codebook {i} and labels: {mi:.4f}")
        ami = adjusted_mutual_info_score(all_cb_index[:, i], labels)
        print(f"Adjusted mutual information between codebook {i} and labels: {ami:.4f}")
        
def analyze_codebook_entropy():
    N = 16
    input_manifest = "data/vq_dasheng_large_cb_16/esc_cuts.jsonl.gz"
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

def analyze_centers():
    N = 16
    embedding_dim = 1280
    
    device = torch.device("cuda")
    quantizer = quantization.Quantizer(
        dim=embedding_dim,
        num_codebooks=N,
        codebook_size=256,
    )
    quantizer_path = "data/quantizer/whisper-turbo-libri-cb-16.pt"
    state_dict = torch.load(quantizer_path)
    if "quantizer" not in state_dict:
        state_dict = {"quantizer": state_dict}
    quantizer.load_state_dict(state_dict["quantizer"])
    quantizer.eval()
    quantizer.to(device)
    
    import pdb; pdb.set_trace()
    centers = quantizer.centers
    mean_per_codebook = centers.mean(dim=1)
    print(centers.shape)


def analyze_speaker():
    N = 16
    input_manifest = "data/vq_whisper_turbo_libri_cb_16/librispeech_cuts_dev-clean.jsonl.gz"
    cuts = load_manifest_lazy(input_manifest)
    
    speakers = cuts.speakers
    for spkr in speakers:
        def filter_speaker(c):
            return c.supervisions[0].speaker == spkr
        import pdb; pdb.set_trace()
        subset = cuts.filter(filter_speaker)
        all_cuts_cb = []
        for cut in subset:
            cb_index = load_codebook_indexes(cut)
            all_cuts_cb.append(cb_index)
    
    
    

if __name__=="__main__":
    # analyze_codebook_entropy()
    # analyze_centers()
    analyze_speaker()
    
