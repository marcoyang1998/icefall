
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

import torch.nn.functional as F
import torchaudio
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition


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
        "--embedding-dir",
        type=str,
        default="data/embeddings"
    )
    
    return parser

def similarity(embed1, embed2, threshold=0.25):
    sim = F.cosine_similarity(embed1, embed2, dim=-1, eps=1e-6)
    return sim, sim > threshold


def test():
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"}
    )
    import pdb; pdb.set_trace()
    signal, fs = torchaudio.load('/star-xy/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac') # spkr1
    embed1 = classifier.encode_batch(signal)

    signal, fs = torchaudio.load('/star-xy/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0001.flac') # spkr1
    embed2 = classifier.encode_batch(signal)

    signal, fs = torchaudio.load('/star-xy/data/LibriSpeech/dev-clean/1462/170138/1462-170138-0000.flac') # spkr2
    embed3 = classifier.encode_batch(signal)

    sim, prediction = similarity(embed1, embed2)
    print(sim, prediction)
    sim, prediction = similarity(embed1, embed3)
    print(sim, prediction)

def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log/log-ecapa-embeddings")
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")
    
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"{params.model_id}-{params.output_manifest}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'{params.model_id}_dev_clean-{rank}.h5'
    else:
        output_manifest = params.embedding_dir / f"{params.model_id}-{params.output_manifest}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'{params.model_id}_dev_clean.h5'
        
    new_cuts = []
    
    # Compute the embeddings and save it
    with NumpyHdf5Writer(embedding_path) as writer:
        logging.info(f"Writing Ecapa embeddings to {embedding_path}")
        for i, cut in enumerate(manifest):
            source = cut.recording.sources[0].source
            audio_input_16khz, _ = torchaudio.load(source)
            audio_input_16khz = audio_input_16khz.to(device)
            
            embeddings = model.encode_batch(audio_input_16khz).detach().to("cpu").numpy()
            new_cut = MonoCut(
                id=cut.id,
                start=cut.start,
                duration=cut.duration,
                channel=cut.channel,
            )
            new_cut.ecapa_embedding = writer.store_array(
                key=cut.id,
                value=embeddings,
            )
            new_cuts.append(new_cut)
            if i and i % 100 == 0:
                logging.info(f"Cuts processed until now: {i}")
    logging.info(f"Finished extracting Ecapa embeddings, processed a total of {i} cuts.")
                
    CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)
    logging.info(f"Saved manifest to {output_manifest}")            
    

def join_manifests(
    input_cuts: CutSet,
    embedding_manifest: str,
    output_dir: str,
):
    # Combine the embedding manifest with the original training manifest
    # by adding a custom field to each cut
    embedding_cuts = load_manifest(embedding_manifest)
    
    assert len(embedding_cuts) == len(input_cuts)
    assert set(input_cuts.ids) == set(embedding_cuts.ids)
    
    embedding_cuts = embedding_cuts.sort_like(input_cuts)
    for cut_idx, (ori_cut, embed_cut) in enumerate(zip(input_cuts, embedding_cuts)):
        assert ori_cut.id == embed_cut.id
        ori_cut.ecapa_embedding = embed_cut.ecapa_embedding
    
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

    params.model_id = "ecapa-tdnn" # for ecapa
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
            mp.spawn(extract_embeddings, args=(splitted_cuts, params), nprocs=nj, join=True)
            manifests =  f"{str(params.embedding_dir)}/{params.model_id}-{params.output_manifest}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {target_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")

    output_manifest = params.input_manifest.replace(".jsonl.gz", "-with-ecapa-embeddings.jsonl.gz")
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=target_manifest,
            output_dir=output_manifest,
        )