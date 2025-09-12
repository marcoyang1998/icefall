import argparse
import logging
import os
from tqdm import tqdm

import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
from lhotse import load_manifest_lazy, CutSet
import multi_quantization as quantization

import umap
import seaborn as sns
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--quantizer-path",
        type=str,
        required=True,
        help="The path to the quantizer model",
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
        help="The manifest to analyse",
    )
    
    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=16,
    )
    
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=1280,
    )
    
    parser.add_argument(
        "--group",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="speaker",
        help="Either speaker or audio"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        required=True,
        default="zipformer-dasheng-mvq-300k",
        help="A small description for the model, used for saving figure"
    )
    
    parser.add_argument(
        "--n-neighbours",
        type=int,
        default=15,
    )
    
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
    )

    return parser.parse_args()

def load_codebook_indexes(c):
    info = c.codebook_indexes
    if isinstance(info, dict):
        filename = info["path"]
        cb_index = np.load(filename)
    else:
        cb_index = c.load_custom("codebook_indexes")
    return cb_index        

def get_esc_category():
    txt_file = "data/esc/categories.txt"
    with open(txt_file, "r") as f:
        data = f.readlines()
    mapping = {}
    for line in data:
        id, name = line.strip().split(" ")
        mapping[int(id)] = name
    return mapping


@torch.no_grad()
def draw_reconstructed_embeddings_audio(args):
    # compute the reconstructed embeddings and visualise them in 2D space
    quantizer_path = args.quantizer_path
    input_manifest = args.input_manifest
    N = args.num_codebooks
    embedding_dim = args.embed_dim
    group = args.group
    suffix = args.suffix
    
    # load the quantizer
    device = torch.device("cuda")
    quantizer = quantization.Quantizer(
        dim=embedding_dim,
        num_codebooks=N,
        codebook_size=256,
    )
    state_dict = torch.load(quantizer_path)
    if "quantizer" not in state_dict:
        state_dict = {"quantizer": state_dict}
    quantizer.load_state_dict(state_dict["quantizer"])
    quantizer.eval()
    quantizer.to(device)
    
    # load the manifest, compute the utterance level embeddings
    class_mapping = get_esc_category()
    
    cuts = load_manifest_lazy(input_manifest)
    
    events = [i for i in range(50)]
    offset = group * 10
    assert offset + 10 <= len(events)
    events = events[offset: offset + 10]
    def filter_cuts(c):
        if int(c.sound_event) in events:
            return True
        return False
    cuts = cuts.filter(filter_cuts)
    
    all_embeddings = []
    labels = []
    for cut in tqdm(cuts):
        cb_index = load_codebook_indexes(cut) # (T,N)
        cb_index = torch.from_numpy(cb_index).to(device)
        embedding = quantizer.decode(cb_index) # (T,C)
        embedding = embedding.mean(dim=0, keepdim=True).cpu() # (1,C)
        
        all_embeddings.append(embedding)
        labels.append(int(cut.sound_event))
        
    all_embeddings = torch.cat(all_embeddings, dim=0) # (T,C)
    
    # visualise the embeddings in 2d space
    logging.info(f"Start fitting the data")
    reducer = umap.UMAP(random_state=42)
    embed_2d = reducer.fit_transform(all_embeddings)
    fig = draw_umap(embed_2d, labels, label_mapping=class_mapping)
    
    figure_name = f"figures/reconstructed_embeddings/esc_{suffix}.png"
    logging.info(f"Saving the figure to {figure_name}")
    fig.savefig(figure_name)
    
@torch.no_grad()
def draw_reconstructed_embeddings_speaker(args):
    # compute the reconstructed embeddings and visualise them in 2D space
    quantizer_path = args.quantizer_path
    input_manifest = args.input_manifest
    N = args.num_codebooks
    embedding_dim = args.embed_dim
    group = args.group
    suffix = args.suffix
    
    # load the quantizer
    device = torch.device("cuda")
    quantizer = quantization.Quantizer(
        dim=embedding_dim,
        num_codebooks=N,
        codebook_size=256,
    )
    state_dict = torch.load(quantizer_path)
    if "quantizer" not in state_dict:
        state_dict = {"quantizer": state_dict}
    quantizer.load_state_dict(state_dict["quantizer"])
    quantizer.eval()
    quantizer.to(device)
    
    # load the manifest, compute the utterance level embeddings
    
    cuts = load_manifest_lazy(input_manifest)
    
    speakers = sorted(list(cuts.speakers))
    num_speakers = min(len(speakers), 10)
    
    # randomly get the first 5 speakers, otherwise visualisation is difficult
    offset = group * 10
    assert offset < len(speakers), "Try smaller group!"
    speakers = speakers[offset: offset + num_speakers]
    
    def filter_speakers(c):
        if c.supervisions[0].speaker in speakers:
            return True
        return False
    cuts = cuts.filter(filter_speakers)
    
    all_embeddings = []
    labels = []
    for cut in tqdm(cuts):
        cb_index = load_codebook_indexes(cut) # (T,N)
        cb_index = torch.from_numpy(cb_index).to(device)
        embedding = quantizer.decode(cb_index) # (T,C)
        embedding = embedding.mean(dim=0, keepdim=True).cpu() # (1,C)
        all_embeddings.append(embedding)
        
        spkr = cut.supervisions[0].speaker
        labels.append(spkr)
        
    all_embeddings = torch.cat(all_embeddings, dim=0) # (T,C)
    
    # visualise the embeddings in 2d space
    logging.info(f"Start fitting the data")
    reducer = umap.UMAP(random_state=42)
    embed_2d = reducer.fit_transform(all_embeddings)
    fig = draw_umap(embed_2d, labels)
    
    figure_name = f"figures/reconstructed_embeddings/vox1_{suffix}.png"
    logging.info(f"Saving the figure to {figure_name}")
    fig.savefig(figure_name)
    
    
@torch.no_grad()
def draw_reconstructed_embeddings_speaker_per_codebook(args):
    # compute the reconstructed embeddings and visualise them in 2D space
    quantizer_path = args.quantizer_path
    input_manifest = args.input_manifest
    N = args.num_codebooks
    embedding_dim = args.embed_dim
    group = args.group
    suffix = args.suffix
    
    # load the quantizer
    device = torch.device("cuda")
    quantizer = quantization.Quantizer(
        dim=embedding_dim,
        num_codebooks=N,
        codebook_size=256,
    )
    state_dict = torch.load(quantizer_path)
    if "quantizer" not in state_dict:
        state_dict = {"quantizer": state_dict}
    quantizer.load_state_dict(state_dict["quantizer"])
    quantizer.eval()
    quantizer.to(device)
    
    # load the manifest, compute the utterance level embeddings
    
    cuts = load_manifest_lazy(input_manifest)
    
    speakers = sorted(list(cuts.speakers))
    num_speakers = min(len(speakers), 20)
    
    # randomly get the first 5 speakers, otherwise visualisation is difficult
    offset = group * 10
    assert offset < len(speakers), "Try smaller group!"
    speakers = speakers[offset: offset + num_speakers]
    speakers = [
        "id10270", "id10276", "id10277", "id10278", "id10272", "id10291", "id10300",
    ]
    
    def filter_speakers(c):
        if c.supervisions[0].speaker in speakers:
            return True
        return False
    cuts = cuts.filter(filter_speakers)
    
    all_embeddings = [[] for n in range(N)]
    labels = []
    for cut in tqdm(cuts):
        cb_index = load_codebook_indexes(cut) # (T,N)
        cb_index = torch.from_numpy(cb_index).to(device).int()
        # embedding = quantizer.decode(cb_index) # (T,C)
        for n in range(N):
            cur_cb_vectors = quantizer.centers[n][cb_index[:, n]] # (T,C)
            cur_embedding = cur_cb_vectors.mean(dim=0, keepdim=True).cpu() #
            all_embeddings[n].append(cur_embedding)
        
        spkr = cut.supervisions[0].speaker
        labels.append(spkr)
        
    for n in range(N):
        all_embeddings[n] = torch.cat(all_embeddings[n], dim=0) # (T,C)
    
    # visualise the embeddings in 2d space
    logging.info(f"Start fitting the data")
    reducer = umap.UMAP(random_state=42, n_neighbors=args.n_neighbours, min_dist=args.min_dist)
    for n, codebook_embedding in enumerate(all_embeddings):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(codebook_embedding)

        embed_2d = reducer.fit_transform(scaled_features)
        fig = draw_umap(embed_2d, labels)
    
        folder = f"figures/reconstructed_embeddings_per_codebook/{suffix}"
        os.makedirs(folder, exist_ok=True)
        figure_name = f"{folder}/vox1-cb-{n}.png"
        logging.info(f"Saving the figure to {figure_name}")
        fig.savefig(figure_name)
    
def draw_umap(X_2d, labels, label_mapping: dict = None):
    # fit the data and draw the 2d scatter plot
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hls", len(unique_labels))

    for i, label in enumerate(unique_labels):
        idx = np.array(labels) == label
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], 
                    label=str(label) if label_mapping is None else label_mapping[label], 
                    alpha=0.7, 
                    s=40, 
                    color=palette[i])
    ax.legend(title="Label", loc="best")
    ax.grid(True)
    fig.tight_layout()
    return fig  
    
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    logging.info(vars(args))
    
    if args.task == "speaker":
        draw_reconstructed_embeddings_speaker_per_codebook(args)
    elif args.task == "audio":
        draw_reconstructed_embeddings_audio(args)
    else:
        raise ValueError()
    