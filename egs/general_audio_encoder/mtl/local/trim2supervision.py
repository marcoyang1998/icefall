import argparse
import logging

from lhotse import load_manifest_lazy

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
    )
    
    parser.add_argument(
        "--output-manifest",
        type=str,
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
     
    input_manifest = args.input_manifest
    output_manifest = args.output_manifest
    
    cuts = load_manifest_lazy(input_manifest)
    cuts = cuts.trim_to_supervisions()
    
    logging.info(f"Saving the manifest to {output_manifest}")
    cuts.to_jsonl(output_manifest)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()    