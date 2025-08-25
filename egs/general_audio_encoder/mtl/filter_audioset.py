from lhotse import load_manifest_lazy

human_files = "human_class.txt"
human_classes = []
with open(human_files, "r") as f:
    data = f.readlines()
    for i, line in enumerate(data):
        if i == 0:
            continue
        cur_class = line.strip().split(",")[0]
        human_classes.append(cur_class)

def filter_audioset(c):
    events = c.supervisions[0].audio_event
    events = events.split(";")
    for event in events:
        if event in human_classes:
            return False
    return True

manifest = "data/fbank_as_ced_mAP50/audioset_cuts_balanced.jsonl.gz"
cuts = load_manifest_lazy(manifest)

non_human_cuts = cuts.filter(filter_audioset)
non_human_cuts = non_human_cuts.drop_supervisions().drop_features()
non_human_cuts.to_jsonl("data/musan/audioset_non_human.jsonl.gz")