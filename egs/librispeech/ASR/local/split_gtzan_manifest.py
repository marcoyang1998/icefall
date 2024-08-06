import random
from lhotse import load_manifest, CutSet

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

def split_cuts(manifest):
    # Split gtzan manifest into 8:1:1 train|dev|test
    cuts = load_manifest(manifest)
    
    train_cuts = []
    dev_cuts = []
    test_cuts = []

    for genre in genres:
        def filter_by_genre(c):
            if c.supervisions[0].genre == genre:
                return True
            else:
                return False
            
        cur_cuts = cuts.filter(filter_by_genre)
        ids = list(range(len(cur_cuts)))
        random.shuffle(ids)
        for i, id in enumerate(ids):
            if i < 80:
                train_cuts.append(cur_cuts[id])
            elif i < 90:
                dev_cuts.append(cur_cuts[id])
            else:
                test_cuts.append(cur_cuts[id])
                
    print(f"A total of {len(train_cuts)} train cuts")
    print(f"A total of {len(dev_cuts)} dev cuts")
    print(f"A total of {len(test_cuts)} test cuts")

    import pdb; pdb.set_trace()
    train_cuts = CutSet.from_cuts(train_cuts)
    train_cuts.to_jsonl("data/fbank_gtzan/cuts_gtzan_train.jsonl.gz")

    dev_cuts = CutSet.from_cuts(dev_cuts)
    dev_cuts.to_jsonl("data/fbank_gtzan/cuts_gtzan_dev.jsonl.gz")

    test_cuts = CutSet.from_cuts(test_cuts)
    test_cuts.to_jsonl("data/fbank_gtzan/cuts_gtzan_test.jsonl.gz")


if __name__=="__main__":
    manifest = "data/fbank_gtzan/cuts_gtzan.jsonl.gz"
    split_cuts(manifest)
