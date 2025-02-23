import os
from lhotse import load_manifest

libri_manifest = "data/manifests/libri_mix_20k_cuts.jsonl.gz"
if not os.path.exists(libri_manifest):
    manifest1 = load_manifest("data/fbank_librispeech/librispeech_cuts_train-clean-100.jsonl.gz")
    manifest2 = load_manifest("data/fbank_librispeech/librispeech_cuts_train-other-500.jsonl.gz")

    manifest1 = manifest1.shuffle().subset(first=10000)
    manifest2 = manifest2.shuffle().subset(first=10000)

    output_manifest = manifest1 + manifest2
    output_manifest = output_manifest.drop_features()
    output_manifest.to_jsonl(libri_manifest)
    print(f"Saved libri manifest to: {libri_manifest}")
else:
    print(f"Libri manifest: {libri_manifest} already exists")
    
giga_manifest = "data/manifests/giga_subset_70k_cuts.jsonl.gz"
if not os.path.exists(giga_manifest):
    manifest_giga = load_manifest("data/fbank_gigaspeech/gigaspeech_cuts_s_raw.jsonl.gz")
    manifest_giga = manifest_giga.shuffle().subset(first=70000)
    manifest_giga = manifest_giga.drop_features()
    manifest_giga.to_jsonl(giga_manifest)
    print(f"Saved giga manifest to: {giga_manifest}")
else:
    print(f"Giga manifest: {giga_manifest} already exists")

aishell_manifest = "data/manifests/aishell_subset_50k_cuts.jsonl.gz"
if not os.path.exists(aishell_manifest):
    manifest_aishell = load_manifest("data/fbank_mtl/aishell_cuts_train.jsonl.gz")
    manifest_aishell = manifest_aishell.shuffle().subset(first=50000)
    manifest_aishell = manifest_aishell.drop_features()
    manifest_aishell.to_jsonl(aishell_manifest)
else:
    print(f"Aishell manifest: {aishell_manifest} already exists")

wenetspeech_manifest = "data/manifests/wenetspeech_subset_trimmed_cuts.jsonl.gz"
if not os.path.exists(wenetspeech_manifest):
    manifest_wenet = load_manifest("data/fbank_wenetspeech_wav_trimmed/wenetspeech_cuts_S.jsonl.gz")
    manifest_wenet = manifest_wenet.shuffle().subset(first=80000)
    manifest_wenet = manifest_wenet.drop_features()
    manifest_wenet.to_jsonl(wenetspeech_manifest)
    print(f"Saved to {wenetspeech_manifest}")
else:
    print(f"Wenet manifest: {wenetspeech_manifest} already exists")
