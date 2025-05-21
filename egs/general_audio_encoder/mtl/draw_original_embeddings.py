import argparse
import logging

import torch
import numpy as np
from lhotse import load_manifest_lazy

import umap
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="audio",
        choices=["audio", "speaker"],
    )
    
    parser.add_argument(
        "--group",
        type=int,
        default=0,
        help="Which groups of data to visualise. Each group consists of 10 sound events or speakers"
    )
    return parser.parse_args()
    

def analyze_speaker(args):
    # draw the utterance level speaker embedding
    suffix = args.suffix
    manifest = args.manifest
    group = args.group
    
    cuts = load_manifest_lazy(manifest)
    
    speakers = sorted(list(cuts.speakers))
    num_speakers = min(len(speakers), 10)
    
    # randomly get the first 5 speakers, otherwise visualisation is difficult
    offset = group * 10
    assert offset < len(speakers), "Try smaller group!"
    speakers = speakers[offset: offset + num_speakers]
    
    labels = []
    all_embeddings = []
    
    def filter_speakers(c):
        if c.supervisions[0].speaker in speakers:
            return True
        return False
    cuts = cuts.filter(filter_speakers)
    
    for i, cut in enumerate(cuts):
        spkr = cut.supervisions[0].speaker
        labels.append(spkr)
        embed  = cut.load_custom("embedding") # (T,C)
        spkr_embed = torch.from_numpy(embed).mean(dim=0, keepdim=True) # (1, C)
        all_embeddings.append(spkr_embed)
        if i % 200 == 0:
            print(f"Finish loading for {i} cuts.")
    
    all_embeddings = torch.cat(all_embeddings).numpy() # (N,C), all speakers embedding
    reducer = umap.UMAP(random_state=42)
    print(f"Start fitting the data")
    all_embeddings_2d = reducer.fit_transform(all_embeddings)
    
    fig = draw_umap(X_2d=all_embeddings_2d, labels=labels)
    output_figure = f"figures/teacher_embeddings/speaker_{suffix}.png"
    logging.info(f"Saving the figure to {output_figure}")
    fig.savefig(output_figure)

def get_esc_category():
    txt_file = "data/esc/categories.txt"
    with open(txt_file, "r") as f:
        data = f.readlines()
    mapping = {}
    for line in data:
        id, name = line.strip().split(" ")
        mapping[int(id)] = name
    return mapping

def analyze_audio_classification(args):
    suffix = args.suffix
    manifest = args.manifest
    group = args.group
    
    cuts = load_manifest_lazy(manifest)
    class_mapping = get_esc_category()
    
    offset = group * 10
    all_events = [i for i in range(50)]
    assert offset < 50
    events = all_events[offset: offset + 10]
    
    def filter_cuts(c):
        if int(c.sound_event) in events:
            return True
        return False

    cuts = cuts.filter(filter_cuts)
    
    labels = []
    all_embeddings = []
    
    for cut in cuts:
        embed  = cut.load_custom("embedding") # (T,C)
        event = int(cut.sound_event)
        labels.append(event)
        audio_embed = torch.from_numpy(embed).mean(dim=0, keepdim=True) # (1, C)
        all_embeddings.append(audio_embed)
    
    all_embeddings = torch.cat(all_embeddings).numpy() # (N,C), all speakers embedding
    reducer = umap.UMAP(random_state=42)
    print(f"Start fitting the data")
    all_embeddings_2d = reducer.fit_transform(all_embeddings)
    
    fig = draw_umap(X_2d=all_embeddings_2d, labels=labels, label_mapping=class_mapping)
    output_figure = f"figures/teacher_embeddings/esc_{suffix}.png"
    logging.info(f"Saving the figure to {output_figure}")
    fig.savefig(output_figure)
    

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
    args = parse_args()
    logging.info(vars(args))
    
    task = args.task
    if task == "speaker":
        analyze_speaker(args)
    elif task == "audio":
        analyze_audio_classification(args)
    else:
        raise ValueError()
    # test_umap()
        
    
        
        
        