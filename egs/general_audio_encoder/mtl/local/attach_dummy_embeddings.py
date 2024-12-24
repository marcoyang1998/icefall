import argparse
from functools import partial
import logging

from zipformer_audio_encoder.utils import _add_dummy_embeddings_and_taskIDs
from lhotse import load_manifest_lazy

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input-manifest",
        type=str,
    )
    
    parser.add_argument(
        "--output-manifest",
        type=str,
    )
    
    return parser.parse_args()

def main():
    args = get_parser()
    logging.info(vars(args))
    
    logging.info(f"Loading {args.input_manifest}")
    cuts = load_manifest_lazy(args.input_manifest)
    
    cuts = cuts.map(partial(_add_dummy_embeddings_and_taskIDs, None)) # do not pass task ID yet
    
    logging.info(f"Saving to {args.output_manifest}")
    cuts.to_jsonl(args.output_manifest)
    
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    main()
