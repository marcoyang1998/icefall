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
        default="speaker",
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
    
    speakers = [
        "2803", "3536", "5338", "3853", "5536", "652", "422", "1673", "2086", "2412"
    ]
    
    labels = []
    all_embeddings = []
    
    max_utt_per_spkr = 25
    speaker_count_dict = {spk: 0 for spk in speakers}
    def filter_speakers(c):
        if c.supervisions[0].speaker in speakers and speaker_count_dict[c.supervisions[0].speaker] < max_utt_per_spkr:
            speaker_count_dict[c.supervisions[0].speaker] += 1
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
    
    output_figure = f"analysis/figures/teacher_embeddings/speaker_{suffix}.pdf"
    logging.info(f"Saving the figure to {output_figure}")
    fig = draw_umap(X_2d=all_embeddings_2d, labels=labels, save_path=output_figure, show_legend=False)
    
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
    
    fig = draw_umap(X_2d=all_embeddings_2d, labels=labels, label_mapping=class_mapping, show_legend=False)
    output_figure = f"figures/teacher_embeddings/esc_{suffix}.png"
    logging.info(f"Saving the figure to {output_figure}")
    fig.savefig(output_figure)
    

def draw_umap(X_2d, labels, label_mapping: dict = None, show_legend: bool = True, save_path: str = None):
    """
    Draw UMAP 2D visualization.

    Args:
        X_2d: numpy array of shape (n_samples, 2)
        labels: list or numpy array of labels
        label_mapping: optional dict for label names
        show_legend: whether to show legend (default: True)
        save_path: optional path to save figure (e.g., 'umap.pdf')
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hls", len(unique_labels))

    for i, label in enumerate(unique_labels):
        idx = np.array(labels) == label
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                   label=str(label) if label_mapping is None else label_mapping[label],
                   alpha=0.7,
                   s=30,
                   color=palette[i])

    if show_legend:
        ax.legend(title="Label", loc="best", fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_path}")

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
        
    
        
        
        