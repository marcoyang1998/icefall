import argparse
import joblib
import logging
import os

import torch
import numpy as np
from lhotse import load_manifest_lazy, CutSet
from lhotse.cut import MonoCut
from lhotse.utils import fastcopy
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from ssl_models import Data2Vec, WavlmModel, HuBERT, W2vBERT
from train_kmeans import normalize_embedding

from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["data2vec", "wavlm", "hubert", "w2v-bert"],
        required=True,
    )
    
    parser.add_argument(
        "--model-version",
        type=str,
    )
    
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="The index starts from 1, so if you want the 12-th layer feature, just set it to 12"
    )
    
    parser.add_argument(
        "--manifest-path",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--output-manifest-path",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--kmeans-model",
        type=str,
        required=True
    )
    
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--weighted-combine", type=str2bool, default=True)
    parser.add_argument("--weight-file", type=str)
    parser.add_argument('--global-mean-file', type=str, required=True)
    parser.add_argument('--global-std-file', type=str, required=True)
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=200,
    )
    
    return parser.parse_args()       

@torch.no_grad()
def collect_tokens(
    args,
    model_name,
    manifest_path,
    kmeans_model_path,
    output_manifest_path,
    weighted_combine,
    layer_idx=21,
    max_duration=200
):
    # loading the pre-trained model
    if model_name == "data2vec":
        model = Data2Vec(model_version=args.model_version)
    elif model_name == "wavlm":
        model = WavlmModel()
    elif model_name == "hubert":
        model = HuBERT(model_version=args.model_version)
    elif model_name == "w2v-bert":
        model = W2vBERT()
    else:
        raise ValueError(f"{model_name} is not supported yet")
    
    model.eval()
    
    device = torch.device("cuda")
    model.to(device)
    
    manifest = load_manifest_lazy(manifest_path)
    dataset = UnsupervisedWaveformDataset(
        manifest
    )
    
    sampler = DynamicBucketingSampler(
        manifest,
        max_duration=max_duration,
        shuffle=False,
        drop_last=False,
    )
    
    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=1,
        persistent_workers=False,
    )
    
    device = torch.device("cuda")
    model.to(device)
    
    if weighted_combine:
        weights = torch.load(args.weight_file).to(device)
        logging.info(f"Using weighted combine: {weights}")
    
    # load the normalization stats
    if args.normalize:
        logging.info("Loading normalization stats")
        global_mean = np.load(args.global_mean_file)
        global_std = np.load(args.global_std_file)
    
    # load the kmeans model
    logging.info(f"Loading kmeans model from {kmeans_model_path}")
    kmeans_model = joblib.load(kmeans_model_path)
    
    new_cuts = []
    count = 0
    # extract the kmeans label
    for i, batch in enumerate(dl):
        cuts = batch["cuts"]
        features, all_hidden_states, embedding_lens = model(batch)
        if weighted_combine:
            all_hidden_states = torch.stack(all_hidden_states, dim=0) # (L,B,T,C)
            all_hidden_states = weights.reshape(-1, 1,1,1) * all_hidden_states # (L,B,T,C) 
            hidden_states = torch.sum(all_hidden_states, dim=0) # (B,T,C)
        else:
            if layer_idx == -1:
                hidden_states = features
            else:
                hidden_states = all_hidden_states[layer_idx] # (B,T,C)
        hidden_states = hidden_states.cpu().numpy()
        
        for j, cut in enumerate(cuts):
            cut = cut if isinstance(cut, MonoCut) else cut.tracks[0].cut
            if args.normalize:
                cur_embeddings = normalize_embedding(
                    hidden_states[j, :embedding_lens[j], :],
                    global_mean,
                    global_std,
                )[0]
            else:
                cur_embeddings = hidden_states[j, :embedding_lens[j], :]
            
            labels = kmeans_model.predict(cur_embeddings)
            
            new_cut = fastcopy(
                cut,
                custom = {"tokens": labels.tolist()},
            )
            new_cuts.append(new_cut)
            count += 1
            if count % 200 == 0:
                logging.info(f"Processed {count} cuts.")
                
    new_cuts = CutSet.from_cuts(new_cuts)
    logging.info(f"Saving the manifest to {output_manifest_path}")
    new_cuts.to_jsonl(output_manifest_path)
            
                
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    
    if os.path.exists(args.output_manifest_path):
        logging.info(f"The manifest {args.output_manifest_path} already exists. Skip this subset.")
    else:
        collect_tokens(
            args,
            model_name=args.model_name,
            manifest_path=args.manifest_path,
            kmeans_model_path=args.kmeans_model,
            output_manifest_path=args.output_manifest_path,
            weighted_combine=args.weighted_combine,
            layer_idx=args.layer_idx,
            max_duration=args.max_duration,
        )