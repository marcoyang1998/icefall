import os

from lhotse import load_manifest_lazy, CutSet

output_folder = "data/vq_whisper_turbo_cb16_firered_zh_en_cb16"
#subsets = ["dev-clean", "dev-other", "train-all-shuf"]
subsets = ["DEV", "L"]

for subset in subsets:
    cuts_whisper = load_manifest_lazy(f"data/vq_whisper_turbo_zh_en_16_v2/wenetspeech_cuts_{subset}.jsonl.gz")
    cuts_firered = load_manifest_lazy(f"data/vq_firered_zh_en_16_v2/wenetspeech_cuts_{subset}.jsonl.gz")
    
    output_manifest = output_folder + f"/wenetspeech_cuts_{subset}.jsonl.gz"
    if os.path.exists(output_manifest):
        print(f"Output manifest exists at {output_manifest}")
        continue

    assert len(cuts_whisper) == len(cuts_firered)

    cuts_firered = cuts_firered.sort_like(cuts_whisper)

    new_cuts = []
    for whisper_cut, firered_cut in zip(cuts_whisper, cuts_firered):
        whisper_cb = whisper_cut.codebook_indexes
        firered_cb = firered_cut.codebook_indexes
        firered_cb.start = whisper_cb.start
        whisper_cut.firered_codebook_indexes = firered_cb
        new_cuts.append(whisper_cut)
    
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest)
    print(f"Saved the manifest to {output_manifest}")