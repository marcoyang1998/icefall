
import argparse
import os
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger
from model import ATST_FrameEncoder

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.dataset import UnsupervisedWaveformDataset, DynamicBucketingSampler
from lhotse.features.io import NumpyHdf5Writer

def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

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
        "--manifest-name",
        type=str,
        required=True,
        help="name of the manifest, e.g embeddings-dev-clean, embeddings-train-clean-100"
    )
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/embeddings"
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--target-manifest-file",
        type=str,
        required=True,
        help="Where to store the manifest augmented with whisper features"
    )
    
    # atst_frame related args
    parser.add_argument(
        "--model-version",
        type=str,
        default="base"
    )
    
    parser.add_argument(
        "--embedding-layer",
        type=str,
        default="-1",
        help="Which layer's representation should be extracted. Indexes are comma separated.",
    )
    
    parser.add_argument(
        "--concat-all-layers",
        type=str2bool,
        default=False,
    )
    
    return parser

@torch.no_grad()
def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log/log-atst_frame-embeddings")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"atst_frame-{params.model_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'atst_frame-{params.model_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}'
    else:
        output_manifest = params.embedding_dir / f"atst_frame-{params.model_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'atst_frame-{params.model_version}-layer-{params.embedding_layer}-{params.manifest_name}'
    
    if params.concat_all_layers:
        logging.info(f"Using all layers feature")
        embedding_layer = None
    else:
        embedding_layer = [int(num.strip()) for num in params.embedding_layer.split(",")]
        logging.info(f"Using embedding from the following layers: {embedding_layer}")
    
    device = torch.device("cuda", rank)
    
    # currently only use the encoder of atst_frame
    logging.info(params)
    model = ATST_FrameEncoder()
    model.to(device)
    model.eval()
    
    logging.info(f"Number of atst_frame encoder params: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Successfully loaded atst_frame model.")
    
    dataset = UnsupervisedWaveformDataset(
        manifest
    )
    
    sampler = DynamicBucketingSampler(
        manifest,
        max_duration=params.max_duration,
        shuffle=False,
        drop_last=False,
    )
    
    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=2,
        persistent_workers=False,
    )
    
    new_cuts = []
    num_cuts = 0
    
    with NumpyHdf5Writer(embedding_path) as writer:
        logging.info(f"Writing atst_frame embeddings to {embedding_path}")
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audios = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            
            embeddings, embedding_lens = model.get_embeddings(
                audio=audios,
                audio_lens=audio_lens,
                layer_idx=embedding_layer,
                concat_all_layers=params.concat_all_layers,
            )
            embeddings = embeddings.detach().to("cpu").numpy()
            
            for idx, cut in enumerate(cuts):    
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.embedding = writer.store_array(
                    key=cut.id,
                    value=embeddings[idx][: embedding_lens[idx]],
                    temporal_dim=0,
                    frame_shift=0.04, # 25 Hz
                    start=cut.start,
                )
                new_cuts.append(new_cut)
                num_cuts += 1
            if i and i % 100 == 0:
                logging.info(f"Cuts processed until now: {num_cuts}")
                
    logging.info(f"Finished extracting atst_frame embeddings, processed a total of {num_cuts} cuts.")
                
    CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)
    logging.info(f"Saved manifest to {output_manifest}")
    
def join_manifests(
    input_cuts: CutSet,
    embedding_manifest: str,
    output_dir: str,
):
    # Combine the teacher embedding manifest with the original manifest for ASR
    embedding_cuts = load_manifest(embedding_manifest)
    
    assert len(embedding_cuts) == len(input_cuts)
    assert set(input_cuts.ids) == set(embedding_cuts.ids)
    
    embedding_cuts = embedding_cuts.sort_like(input_cuts)
    for cut_idx, (ori_cut, embed_cut) in enumerate(zip(input_cuts, embedding_cuts)):
        assert ori_cut.id == embed_cut.id
        ori_cut.embedding = embed_cut.embedding
    
    input_cuts.to_jsonl(output_dir)
    print(f"Saved the joined manifest to {output_dir}")
    
def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 29.9:
        return False
    return True

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    params.embedding_dir = Path(params.embedding_dir)
    
    nj = params.num_jobs
    cuts = load_manifest(params.input_manifest)
    cuts = cuts.filter(remove_short_and_long_utt) # remove audio longer than 30s
    print(f"Finished loading manifest")
    
    embedding_manifest = params.embedding_dir / f"atst_frame-{params.model_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
    
    if not embedding_manifest.exists():
        if nj == 1:
            extract_embeddings(
                rank=0,
                manifest=cuts,
                params=params,    
            )
        else:
            splitted_cuts = cuts.split(num_splits=nj)
            print(f"Finished splitting manifest")
            mp.spawn(extract_embeddings, args=(splitted_cuts, params), nprocs=nj, join=True)
            manifests =  f"{str(params.embedding_dir)}/atst_frame-{params.model_version}-layer-{params.embedding_layer}-{params.manifest_name}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {embedding_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.target_manifest_file
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=embedding_manifest,
            output_dir=output_manifest,
        )
    
    