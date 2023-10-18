
import argparse
import os
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger

import torch
import torch.multiprocessing as mp
import torchaudio
from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer

from BEATs import BEATs, BEATsConfig
from Tokenizers import TokenizersConfig, Tokenizers

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--output-manifest",
        type=str,
        required=True,
        help="name of the manifest, e.g embeddings-dev-clean"
    )
    
    parser.add_argument(
        "--beats-ckpt",
        type=str,
        default="data/models/BEATs/BEATs_iter3_plus_AS2M.pt",
    )
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/embeddings"
    )
    
    return parser
    

def tokenize():
    checkpoint = torch.load('data/models/BEATs/Tokenizer_iter3_plus_AS20K.pt')

    cfg = TokenizersConfig(checkpoint['cfg'])
    BEATs_tokenizer = Tokenizers(cfg)
    BEATs_tokenizer.load_state_dict(checkpoint['model'])
    BEATs_tokenizer.eval()
    BEATs_tokenizer.to("cuda")


    # tokenize the audio and generate the labels
    audio_input_16khz, _ = torchaudio.load('/star-xy/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac')
    padding_mask = torch.zeros_like(audio_input_16khz).bool()

    import pdb; pdb.set_trace()
    labels = BEATs_tokenizer.extract_labels(audio_input_16khz, padding_mask=padding_mask)
    print(labels)
    
def get_embeddings():
    checkpoint = torch.load('data/models/BEATs/BEATs_iter3_plus_AS2M.pt')

    cfg = BEATsConfig(checkpoint['cfg'])
    
    device = torch.device("cuda")
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    BEATs_model.to(device)

    # extract the the audio representation
    audio_input_16khz, _ = torchaudio.load('/star-xy/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac')
    audio_input_16khz = audio_input_16khz.to(device)
    padding_mask = torch.zeros_like(audio_input_16khz).bool().to(device)
    
    representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
    print(representation.shape)

def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log/log-beats-embeddings")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"{params.model_id}-{params.output_manifest}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'{params.model_id}-{params.output_manifest}-{rank}.h5'
    else:
        output_manifest = params.embedding_dir / f"{params.model_id}-{params.output_manifest}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'{params.model_id}-{params.output_manifest}.h5'
    
    logging.info(params)
    
    checkpoint = torch.load(params.beats_ckpt)
    cfg = BEATsConfig(checkpoint['cfg'])
    logging.info(f"Successfully load BEATs model.")
    
    device = torch.device("cuda")
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    BEATs_model.to(device)    
    
    new_cuts = []
    
    with NumpyHdf5Writer(embedding_path) as writer:
        logging.info(f"Writing BEATs embeddings to {embedding_path}")
        for i, cut in enumerate(manifest):
            source = cut.recording.sources[0].source
            audio_input_16khz, _ = torchaudio.load(source)
            audio_input_16khz = audio_input_16khz.to(device)
            padding_mask = torch.zeros_like(audio_input_16khz).bool().to(device)
            
            embeddings = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0].detach().to("cpu").numpy()
            new_cut = MonoCut(
                id=cut.id,
                start=cut.start,
                duration=cut.duration,
                channel=cut.channel,
            )
            new_cut.beats_embedding = writer.store_array(
                key=cut.id,
                value=embeddings,
            )
            new_cuts.append(new_cut)
            if i and i % 100 == 0:
                logging.info(f"Cuts processed until now: {i}")
    logging.info(f"Finished extracting BEATs embeddings, processed a total of {i} cuts.")
                
    CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)
    logging.info(f"Saved manifest to {output_manifest}")
    
def join_manifests(
    input_cuts: CutSet,
    embedding_manifest: str,
    output_dir: str,
):
    embedding_cuts = load_manifest(embedding_manifest)
    
    assert len(embedding_cuts) == len(input_cuts)
    assert set(input_cuts.ids) == set(embedding_cuts.ids)
    
    embedding_cuts = embedding_cuts.sort_like(input_cuts)
    for cut_idx, (ori_cut, embed_cut) in enumerate(zip(input_cuts, embedding_cuts)):
        assert ori_cut.id == embed_cut.id
        ori_cut.beats_embedding = embed_cut.beats_embedding
    
    input_cuts.to_jsonl(output_dir)
    print(f"Saved the joined manifest to {output_dir}")


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
    print(f"Finished loading manifest")
    
    params.model_id = params.beats_ckpt.split("/")[-1].replace(".pt", "")
    target_manifest = params.embedding_dir / f"{params.model_id}-{params.output_manifest}.jsonl.gz"
    
    if not target_manifest.exists():
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
            manifests =  f"{str(params.embedding_dir)}/{params.model_id}-{params.output_manifest}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {target_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.input_manifest.replace(".jsonl.gz", "-with-beats-embeddings.jsonl.gz")
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=target_manifest,
            output_dir=output_manifest,
        )
    
    
    
        
    
        
        
        
        