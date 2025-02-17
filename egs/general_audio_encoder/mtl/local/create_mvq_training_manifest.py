import os
from lhotse import load_manifest

english_manifest = "data/manifests/libri_giga_mix_cuts.jsonl.gz"
if not os.path.exists(english_manifest):
    manifest1 = load_manifest("data/fbank_librispeech/librispeech_cuts_train-clean-100.jsonl.gz")
    manifest2 = load_manifest("data/fbank_librispeech/librispeech_cuts_train-other-500.jsonl.gz")

    manifest1 = manifest1.shuffle().subset(first=7500)
    manifest2 = manifest2.shuffle().subset(first=7500)

    output_manifest = manifest1 + manifest2
    output_manifest = output_manifest.drop_features()
    output_manifest.to_jsonl(english_manifest)
else:
    print(f"English manifest: {english_manifest} already exists")

aishell_manifest = "data/manifests/aishell_subset_cuts.jsonl.gz"
if not os.path.exists(aishell_manifest):
    manifest_aishell = load_manifest("data/fbank_mtl/aishell_cuts_train.jsonl.gz")
    manifest_aishell = manifest_aishell.shuffle().subset(first=40000)
    manifest_aishell = manifest_aishell.drop_features()
    manifest_aishell.to_jsonl(aishell_manifest)
else:
    print(f"Aishell manifest: {aishell_manifest} already exists")

wenetspeech_manifest = "data/manifests/wenetspeech_subset_cuts.jsonl.gz"
if not os.path.exists(wenetspeech_manifest):
    manifest_wenet = load_manifest("data/fbank_wenetspeech/wenetspeech_cuts_S.jsonl.gz")
    manifest_wenet = manifest_wenet.shuffle().subset(first=80000)
    manifest_wenet = manifest_wenet.drop_features()
    manifest_wenet.to_jsonl(wenetspeech_manifest)
    print(f"Saved to {wenetspeech_manifest}")
else:
    print(f"Wenet manifest: {wenetspeech_manifest} already exists")
