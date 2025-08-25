from lhotse import load_manifest_lazy

cuts = load_manifest_lazy("data/fbank/musan_cuts.jsonl.gz")

def filter_speech(c):
    if c.id.startswith("speech"):
        return True
    return False

def filter_non_speech(c):
    if not c.id.startswith("speech"):
        return True
    return False

speech_cuts = cuts.filter(filter_speech)
speech_cuts.to_jsonl("data/musan/musan_speech.jsonl.gz")

non_speech_cuts = cuts.filter(filter_non_speech)
non_speech_cuts.to_jsonl("data/musan/musan_non_speech.jsonl.gz")