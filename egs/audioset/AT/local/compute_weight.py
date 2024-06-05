import lhotse
from lhotse import load_manifest

audioset_balanced_cuts = "data/fbank/cuts_audioset_balanced.jsonl.gz"
audioset_unbalanced_cuts = "data/fbank/cuts_audioset_unbalanced.jsonl.gz"
audioset_balanced_cuts = load_manifest(audioset_balanced_cuts)
audioset_unbalanced_cuts = load_manifest(audioset_unbalanced_cuts)

cuts = lhotse.combine(audioset_balanced_cuts, audioset_unbalanced_cuts)

print(f"A total of {len(cuts)} cuts.")

label_count = [0] * 527 # a total of 527 classes
for c in cuts:
    audio_event = c.supervisions[0].audio_event
    labels = list(map(int, audio_event.split(";")))
    for label in labels:
        label_count[label] += 1

print([(i, label_count[i]) for i in range(527)])

with open("data/fbank/sample_weights_full.txt", "w") as f:
    for c in cuts:
        audio_event = c.supervisions[0].audio_event
        labels = list(map(int, audio_event.split(";")))
        weight = 0
        for label in labels:
            weight += 1000 / (label_count[label] + 0.01)
        f.write(f"{c.id} {weight}\n")
        

