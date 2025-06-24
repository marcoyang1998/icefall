from lhotse import load_manifest_lazy

cut_set = load_manifest_lazy("data_s3/fbank_librispeech/librispeech_cuts_train-all-shuf.jsonl.gz")

cut_set = (
    cut_set
    + cut_set.perturb_speed(0.9)
    + cut_set.perturb_speed(1.1)
)

cut_set.to_jsonl("data_s3/librispeech_perturbed/librispeech_cuts_train-all.jsonl.gz")