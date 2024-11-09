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

from dasheng import dasheng_base, dasheng_06B, dasheng_12B
from train_kmeans import normalize_embedding


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
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
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=200,
    )
    
    return parser.parse_args()       

@torch.no_grad()
def collect_tokens(
    model_name,
    model_version,
    manifest_path,
    kmeans_model_path,
    output_manifest_path,
    layer_idx=21,
    max_duration=200
):
    # loading the pre-trained model
    if model_name == "dasheng":
        if model_version == "base":
            model = dasheng_base()
        elif model_version == "medium":
            model = dasheng_06B()
        elif model_version == "large":
            model = dasheng_12B()
        else:
            raise ValueError(f"{model_name} do not have {model_version}")
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
    
    # load the normalization stats
    logging.info("Loading normalization stats")
    if args.model_name == "dasheng":
        global_mean = np.load(f"normalization_stats/dasheng-{args.model_version}-mu.npy")
        global_std = np.load(f"normalization_stats/dasheng-{args.model_version}-std.npy")
    else:
        raise ValueError(f"{model_name} is not supported yet")
    
    # load the kmeans model
    logging.info(f"Loading kmeans model from {kmeans_model_path}")
    kmeans_model = joblib.load(kmeans_model_path)
    
    new_cuts = []
    count = 0
    # extract the kmeans label
    for i, batch in enumerate(dl):
        cuts = batch["cuts"]
        audio = batch["audio"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        if layer_idx == -1:
            features, _ = model(audio)
            embeddings = features
        else:
            features, all_hidden_states = model(audio, output_hidden_states=True)
            embeddings = all_hidden_states[layer_idx]
            
        embeddings = embeddings.cpu().numpy()
        embedding_lens = audio_lens // 640 # The output frequency is 25Hz
        
        for j, cut in enumerate(cuts):
            cut = cut if isinstance(cut, MonoCut) else cut.tracks[0].cut
            cur_embeddings = normalize_embedding(
                embeddings[j, :embedding_lens[j], :],
                global_mean,
                global_std,
            )[0]
            
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
            model_name=args.model_name,
            model_version=args.model_version,
            manifest_path=args.manifest_path,
            kmeans_model_path=args.kmeans_model,
            output_manifest_path=args.output_manifest_path,
            layer_idx=args.layer_idx,
            max_duration=args.max_duration,
        )