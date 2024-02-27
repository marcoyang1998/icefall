import argparse
import csv
import logging

from lhotse import load_manifest_lazy

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--num-references",
        type=int,
        default=5,
    )
    
    return parser

def main():
    # Write out the reference captions into a csv file conforming the evaluation format
    # Can be used for both audiocaps and clotho
    parser = get_parser()
    args = parser.parse_args()
    
    input_manifest = args.input_manifest
    output_csv = args.output_csv
    num_references = args.num_references
    
    cuts = load_manifest_lazy(input_manifest)
    header = [f"caption_reference_0{i}" for i in range(1,num_references+1)]
    header = ["file_name"] + header
    
    with open(output_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        
        for c in cuts:
            ref_captions = c.supervisions[0].audio_captions.split(";;")
            assert len(ref_captions) == num_references
            w.writerow([c.id] + ref_captions)
            
    logging.info("Finished")
            
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
