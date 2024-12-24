import argparse
import logging

from lhotse import CutSet, load_manifest
from lhotse.utils import fastcopy

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--orig-manifest",
        type=str,
    )
    
    parser.add_argument(
        "--mvq-manifest",
        type=str,
    )
    
    parser.add_argument(
        "--output-manifest",
        type=str,
    )
    return parser.parse_args()

def remove_sp(c):
    if "sp0.9" in c.id or "sp1.1" in c.id:
        return False
    return True

def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 30.0:
        return False
    return True
    

def main(args):
    orig_manifest = load_manifest(args.orig_manifest)
    mvq_manifest = load_manifest(args.mvq_manifest)
    
    mvq_manifest = mvq_manifest.filter(remove_sp)
    orig_manifest = orig_manifest.filter(remove_short_and_long_utt).filter(remove_sp)
    assert len(mvq_manifest) == len(orig_manifest)
    # mvq_manifest = mvq_manifest.sort_like(orig_manifest)
    
    mvq_dict = {}
    for c in mvq_manifest:
        key = c.id.rsplit("-", 1)[0]
        mvq_dict[key] = c.codebook_indexes
    
    new_cuts = []
    for c in orig_manifest:
        key = c.id.rsplit("-", 1)[0]
        codebook_indexes = mvq_dict[key]
        new_cut = fastcopy(
            c,
            custom={"codebook_indexes": codebook_indexes}
        )
        new_cuts.append(new_cut)
    
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(args.output_manifest)
    logging.info(f"Saved manfiest to: {args.output_manifest}")
    
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    logging.info(vars(args))
    main(args)
    