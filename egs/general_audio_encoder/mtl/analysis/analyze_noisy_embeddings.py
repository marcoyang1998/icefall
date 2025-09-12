
import argparse
import os
import logging
from pathlib import Path
from tqdm import tqdm

from wavlm_model import WavlmModel
from draw_original_embeddings import draw_umap
from icefall.utils import AttributeDict, setup_logger

import torch
from lhotse import load_manifest_lazy, load_manifest
import numpy as np

import umap
import seaborn as sns
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--speaker-group",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--embedding-layer",
        type=int,
        default=21,
        help="Which layer's representation should be extracted, index start from 1"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
    )

    # wavlm related args
    parser.add_argument(
        "--wavlm-version",
        type=str,
        default="large"
    )
    
    return parser

def analyze_noisy_embeddings(args):
    from lhotse import load_manifest_lazy, load_manifest
    
    params = AttributeDict()
    params.update(vars(args))
    input_manifest = args.input_manifest
    group = args.speaker_group
    embedding_layer = args.embedding_layer
    
    # prepare the input manifest
    cuts = load_manifest(input_manifest)
    
    speakers = sorted(list(cuts.speakers))
    num_speakers = min(len(speakers), 20)
    
    # randomly get the first 5 speakers, otherwise visualisation is difficult
    offset = group * 10
    assert offset < len(speakers), "Try smaller group!"
    speakers = speakers[offset: offset + num_speakers]
    # speakers = [
    #     "id10270", "id10276", "id10277", "id10278", "id10272", "id10291", "id10300",
    # ]
    
    def filter_speakers(c):
        if c.supervisions[0].speaker in speakers:
            return True
        return False
    cuts = cuts.filter(filter_speakers)
    
    # get the model
    model = WavlmModel(model_version=args.wavlm_version)
    model.eval()
    model = model.to("cuda")
    
    all_clean_embeddings = []
    all_noisy_embeddings = []
    all_labels = []
    for cut in tqdm(cuts):
        audio = cut.load_audio()[0] # noisy audio
        clean_audio = cut.tracks[0].cut.load_audio()[0]
        audios = np.stack([audio, clean_audio], axis=0)
        batch = {
            "audio": torch.from_numpy(audios),
            "audio_lens": torch.tensor([audio.shape[0]] * 2),
        }
        
        embeddings, embedding_lens = model.get_embeddings(
            batch=batch,
            layer_idx=embedding_layer # which layer's embedding to be stored
        )
        assert embeddings.shape[0] == 2
        embeddings = embeddings.detach().to("cpu")
        avg_embedding = embeddings.mean(dim=1) # (2,C)
        all_noisy_embeddings.append(avg_embedding[0:1])
        all_clean_embeddings.append(avg_embedding[1:2])
        
        spkr = cut.supervisions[0].speaker
        all_labels.append(spkr)
        
    embeddings = [all_clean_embeddings, all_noisy_embeddings]
    names = ["clean", "noisy"]
    for name, all_embeddings in zip(names, embeddings):
        all_embeddings = torch.cat(all_embeddings).numpy() # (N,C), all speakers embedding
        reducer = umap.UMAP(random_state=42)
        print(f"Start fitting the data")
        all_embeddings_2d = reducer.fit_transform(all_embeddings)
        
        fig = draw_umap(X_2d=all_embeddings_2d, labels=all_labels)
        output_figure = f"analysis/figures/vox1-{params.suffix}-{name}.png"
        logging.info(f"Saving the figure to {output_figure}")
        fig.savefig(output_figure)

    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    analyze_noisy_embeddings(args)