import argparse
import logging

from lhotse import load_manifest

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--source-manifest", type=str, required=True)
    parser.add_argument("--noise-manifest", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--snr", type=int, required=True)
    
    return parser.parse_args()

def mix_cuts(args):
    source_manifest = args.source_manifest
    noise_manifest = args.noise_manifest
    output_manifest = args.output_manifest
    snr = args.snr
        
    orig_cuts = load_manifest(source_manifest)
    orig_cuts = orig_cuts.drop_features() # we mix at waveform level
    
    noise_cuts = load_manifest(noise_manifest)
    noise_cuts = noise_cuts.drop_features()

    logging.info(f"Start mixing the cuts with SNR={snr} dB.")
    mixed_cuts = orig_cuts.mix(noise_cuts, preserve_id="left", snr=snr)
    
    logging.info(f"Saving the mixed cuts to {output_manifest}.")
    mixed_cuts.to_jsonl(output_manifest)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    
    mix_cuts(args)

    