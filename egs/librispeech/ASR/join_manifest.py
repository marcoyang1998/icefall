import argparse
import logging

from lhotse import CutSet, load_manifest

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--recording-manifest",
        type=str,   
    )
    
    parser.add_argument(
        "--supervision-manifest",
        type=str,
    )
    
    parser.add_argument(
        "--output-manifest",
        type=str,
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    recording = load_manifest(args.recording_manifest)
    supervision = load_manifest(args.supervision_manifest)
    
    cuts = CutSet.from_manifests(
        recordings=recording,
        supervisions=supervision
    )
    
    logging.info(f"Saving manifest to {args.output_manifest}")
    cuts.to_jsonl(args.output_manifest)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()