import argparse
import logging
import os

import torch
from lhotse import load_manifest_lazy, CutSet
from lhotse.features.io import LilcomChunkyWriter
from lhotse.utils import fastcopy
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from dasheng import dasheng_base, dasheng_06B, dasheng_12B
from icefall.utils import str2bool

MODEL_DICT = {
    "base": dasheng_base,
    "medium": dasheng_06B,
    "large": dasheng_12B,
}

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dasheng-version",
        type=str,
        required=True,
        choices=["base", "medium", "large"]
    )
    
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="""Index starts from 1, to get the 12-th layer features, set layer-idx=12. If 
        layer-idx is set to -1, then use the final output feature (after final norm).
        """
    )
    
    parser.add_argument(
        "--weighted-combine",
        type=str2bool,
        default=False,
        help="Whether to use weighted combination of features from different layers."
    )
    
    parser.add_argument(
        "--weight-file",
        type=str,
        default="Path to the file containing the layer weights."
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=200,
    )
    
    return parser.parse_args()

def test_dasheng():
    import torchaudio
    model = dasheng_base()

    model = model.eval()

    wav = "download/audioset/balanced/ZzyWbehtt0M_30.000.wav"
    audio, fs = torchaudio.load(wav)

    with torch.no_grad():
        import pdb; pdb.set_trace()
        features, all_hidden_states = model(audio, output_hidden_states=True) # 25 Hz output, 

    import pdb; pdb.set_trace()
    print(features.shape) #(B,T,C)
    
def combine_embeddings(all_hidden_states, weight=None):
    all_hidden_states = torch.stack(all_hidden_states, dim=0) # (L,B,T,C)
    all_hidden_states = weight.reshape(-1, 1,1,1) * all_hidden_states # (L,B,T,C) 
    hidden_states = torch.sum(all_hidden_states, dim=0) # (B,T,C)
    return hidden_states
    
@torch.no_grad()
def collect_results(
    dasheng_version,
    manifest_path,
    embedding_path,
    weighted_combine,
    weight_file,
    output_manifest_path,
    layer_idx=21,
    max_duration=200
):
    
    # load the pretrained dasheng model
    model = MODEL_DICT[dasheng_version]()
    model.eval()
    
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
        logging.info(f"Loading weight from: {weight_file}")
        weight = torch.load(weight_file).to(device)
        logging.info(f"Loaded weights: {weight}")
    else:
        weight = None
    
    new_cuts = []
    num_cuts = 0
    with LilcomChunkyWriter(embedding_path) as writer:
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio_input_16khz = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            
            features, all_hidden_states = model(audio_input_16khz, output_hidden_states=True)
            
            if weighted_combine:
                layer_results = combine_embeddings(all_hidden_states, weight=weight)
            else:
                if layer_idx != -1:
                    layer_results = all_hidden_states[layer_idx] # (B,T,C)
                else:
                    layer_results = features # (B,T,C)
                    
            layer_results = layer_results.cpu().numpy()
            embedding_lens = audio_lens // 640 # The output frequency is 25Hz
            
            for j, cut in enumerate(cuts):
                embeddings = writer.store_array(
                    key=cut.id,
                    value=layer_results[j][:embedding_lens[j]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=0,
                )
                new_cut = fastcopy(
                    cut,
                    custom={"dasheng_embedding": embeddings}
                )
                new_cuts.append(new_cut)
                num_cuts += 1
                if num_cuts and num_cuts % 200 == 0:
                    logging.info(f"Cuts processed until now: {num_cuts}")
    
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest_path)
    logging.info(f"Manifest saved to {output_manifest_path}")
            
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    dasheng_version = args.dasheng_version
    layer_idx = args.layer_idx
    subset = args.subset
    weighted_combine = args.weighted_combine
    weight_file = args.weight_file
    
    manifest_path = f"data/fbank_audioset/cuts_audioset_{subset}.jsonl.gz"
    if weighted_combine:
        embedding_path = f"embeddings/dasheng_embeddings/dasheng-{dasheng_version}-weighted-combine-{subset}.h5"
        output_manifest_path = f"manifests/{subset}-dasheng-{dasheng_version}-weighted-combine.jsonl.gz"
    else:
        embedding_path = f"embeddings/dasheng_embeddings/dasheng-{dasheng_version}-layer-{layer_idx}-{subset}.h5"
        output_manifest_path = f"manifests/{subset}-dasheng-{dasheng_version}-layer-{layer_idx}.jsonl.gz"
    
    if not os.path.exists(output_manifest_path):
        collect_results(
            dasheng_version=dasheng_version,
            manifest_path=manifest_path,
            embedding_path=embedding_path,
            output_manifest_path=output_manifest_path,
            weighted_combine=weighted_combine,
            weight_file=weight_file,
            layer_idx=layer_idx,
            max_duration=args.max_duration,
        )
    else:
        logging.info(f"The manifest {output_manifest_path} already exists. Skip this subset.")