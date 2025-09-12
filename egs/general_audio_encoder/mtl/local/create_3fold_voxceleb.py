import logging
from lhotse import load_manifest_lazy, CutSet


def create_3fold_voxceleb(manifest, output_manifest):
    cut_set = load_manifest_lazy(manifest).drop_features()
    cut_set_sp09 = cut_set.perturb_speed(0.9)
    def modify_speaker(c):
        c.supervisions[0].speaker = c.supervisions[0].speaker + "_sp09"
        return c
    cut_set_sp09 = cut_set_sp09.map(modify_speaker)
    
    cut_set_sp11 = cut_set.perturb_speed(1.1)
    def modify_speaker(c):
        c.supervisions[0].speaker = c.supervisions[0].speaker + "_sp11"
        return c
    cut_set_sp11 = cut_set_sp11.map(modify_speaker)
    
    all_cuts = CutSet.mux(
        *[cut_set, cut_set_sp09, cut_set_sp11],
        weights=[1.0, 1.0, 1.0],
        stop_early=False
    )
    all_cuts.to_jsonl(output_manifest)
    print(f"Saved to {output_manifest}")
    
if __name__ == "__main__":
    import logging
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    subsets = ["vox1", "vox2"]
    for subset in subsets:
        manifest = f"data/fbank_voxceleb/{subset}_cuts_dev.jsonl.gz"
        output_manifest = f"data/fbank_voxceleb/{subset}_cuts_dev_3fold.jsonl.gz"
        create_3fold_voxceleb(manifest, output_manifest)