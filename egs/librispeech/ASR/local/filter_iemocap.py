from lhotse import load_manifest_lazy
from lhotse import CutSet

subset = "train"
cuts = load_manifest_lazy(f"data/fbank_iemocap/cuts_iemocap_{subset}.jsonl.gz")

def change_exc(c):
    if c.supervisions[0].emotion == "exc":
        c.supervisions[0].emotion = "hap"
    return c

def filter_iemocap(c):
    if c.supervisions[0].emotion not in ["hap", "ang", "sad", "neu"]:
        return False
    return True

cuts = cuts.map(change_exc)
cuts = cuts.filter(filter_iemocap)

emo_dict = {}

for cut in cuts:
    emo = cut.supervisions[0].emotion
    if emo in emo_dict:
        emo_dict[emo] += 1
    else:
        emo_dict[emo] = 1

print(emo_dict)
cuts.to_jsonl(f"data/fbank_iemocap/cuts_iemocap_{subset}_filtered.jsonl.gz")

