from lhotse import load_manifest

manifest1 = load_manifest("data/manifests/librispeech_cuts_train-clean-100.jsonl.gz")
manifest2 = load_manifest("data/manifests/librispeech_cuts_train-other-500.jsonl.gz")

manifest1 = manifest1.shuffle().subset(first=7500)
manifest2 = manifest2.shuffle().subset(first=7500)

out_manifest = manifest1 + manifest2

out_manifest.to_jsonl(f"data/manifests/librispeech_cuts_mix.jsonl.gz")



