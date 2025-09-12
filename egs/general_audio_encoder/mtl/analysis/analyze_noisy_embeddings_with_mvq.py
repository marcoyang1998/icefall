
import argparse
import os
import logging
from pathlib import Path
from tqdm import tqdm

from wavlm_model import WavlmModel
from draw_original_embeddings import draw_umap
from icefall.utils import AttributeDict, setup_logger


import torch
from lhotse import load_manifest_lazy, load_manifest, CutSet
import numpy as np
import multi_quantization as quantization

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
    
    # quantizer related
    parser.add_argument(
        "--quantizer-path",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=16,
    )

    # wavlm related args
    parser.add_argument(
        "--wavlm-version",
        type=str,
        default="large"
    )
    
    parser.add_argument(
        "-embedding-dim",
        type=int,
        default=1024,
    )
    
    return parser

def normalize_data(data, mean, std):
    return (data - mean) / std

def get_codebook_indexes(quantizer, embeddings, mu, std):
    normalized_embed = normalize_data(embeddings, mu, std)
    codebook_indexes = quantizer.encode(normalized_embed)
    codebook_indexes = codebook_indexes.detach().to("cpu").numpy()
    return codebook_indexes

def generate_manifest(cuts, speaker_group):
    speakers = sorted(list(cuts.speakers))
    num_speakers = min(len(speakers), 10)
    
    # randomly get the first 5 speakers, otherwise visualisation is difficult
    offset = speaker_group * 10
    assert offset < len(speakers), "Try smaller group!"
    speakers = speakers[offset: offset + num_speakers]
    
    # speakers = [
    #     "id10270", "id10276", "id10277", "id10278", "id10272", "id10291", "id10300",
    # ]
    
    # speakers = [
    #     "id10276", "id10277", "id10278", "id10272", "id10291", "id10300",
    # ]
    
    new_cuts = CutSet()
    for speaker in speakers:
        def filter_speakers(c):
            if c.supervisions[0].speaker == speaker:
                return True
            return False
        cur_speaker_cuts = cuts.filter(filter_speakers)
        cur_speaker_cuts = cur_speaker_cuts.subset(first=20)
        new_cuts += cur_speaker_cuts
        
    return new_cuts
    
    
def load_quantizer(params):
    
    quantizer = quantization.Quantizer(
        dim=params.embedding_dim,
        num_codebooks=params.num_codebooks,
        codebook_size=256,
    )
    state_dict = torch.load(params.quantizer_path)
    if "quantizer" not in state_dict:
        state_dict = {"quantizer": state_dict}
    quantizer.load_state_dict(state_dict["quantizer"])
    quantizer.eval()
    return quantizer, state_dict

def analyze_noisy_embeddings(args):
    from lhotse import load_manifest_lazy, load_manifest
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    params = AttributeDict()
    params.update(vars(args))
    input_manifest = args.input_manifest
    group = args.speaker_group
    embedding_layer = args.embedding_layer
    
    # prepare the input manifest
    cuts = load_manifest(input_manifest)

    cuts = generate_manifest(cuts, group)
    
    # get the model
    model = WavlmModel(model_version=args.wavlm_version)
    model.eval()
    model = model.to(device)
    
    # get the quantizer
    quantizer, quantizer_states = load_quantizer(params)
    quantizer.to(device)
    mu = quantizer_states["mean"].to(device)
    std = quantizer_states["std"].to(device)
    
    all_clean_embeddings = []
    all_noisy_embeddings = []
    all_clean_embeddings_recon = []
    all_noisy_embeddings_recon = []
    one_cb_clean_embeddings_recon = []
    one_cb_noisy_embeddings_recon = []
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
        
        # wavlm embedding
        embeddings = embeddings.detach().to("cpu")
        avg_embedding = embeddings.mean(dim=1) # (2,C)
        all_noisy_embeddings.append(avg_embedding[0:1])
        all_clean_embeddings.append(avg_embedding[1:2])
        
        # encode the embedding
        codebook_indexes = get_codebook_indexes(quantizer, embeddings.to(device), mu, std)
        assert np.min(codebook_indexes) >= 0
        assert np.max(codebook_indexes) < 256
        cb_index = torch.from_numpy(codebook_indexes).to(device)
        recon_embeddings = quantizer.decode(cb_index).detach().to("cpu")
        recon_avg_embedding = recon_embeddings.mean(dim=1) # (2,C)
        all_noisy_embeddings_recon.append(recon_avg_embedding[0:1])
        all_clean_embeddings_recon.append(recon_avg_embedding[1:2])
        
        # encode the per-codebook embedding
        n=6
        one_cb_noisy_vectors = quantizer.centers[n][cb_index[0, :, n].unsqueeze(1).int()]
        one_cb_clean_vectors = quantizer.centers[n][cb_index[1, :, n].unsqueeze(1).int()]
        one_cb_noisy_vectors = one_cb_noisy_vectors.mean(dim=0).detach().cpu()
        one_cb_clean_vectors = one_cb_clean_vectors.mean(dim=0).detach().cpu()
        one_cb_noisy_embeddings_recon.append(one_cb_noisy_vectors)
        one_cb_clean_embeddings_recon.append(one_cb_clean_vectors)

        # collect speaker label
        spkr = cut.supervisions[0].speaker
        all_labels.append(spkr)
        
    embeddings = [
        all_clean_embeddings,
        all_noisy_embeddings,
        all_clean_embeddings_recon,
        all_noisy_embeddings_recon,
        one_cb_clean_embeddings_recon,
        one_cb_noisy_embeddings_recon,
    ]
    names = ["wavlm-clean", "wavlm-noisy", "mvq-clean", "mvq-noisy", "cb-6-clean", "cb-6-noisy"]
    for name, all_embeddings in zip(names, embeddings):
        all_embeddings = torch.cat(all_embeddings).numpy() # (N,C), all speakers embedding
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
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