import argparse
import csv
from collections import defaultdict
import gzip
import logging
import os
from pathlib import Path
from tqdm import tqdm


from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.audio.utils import AudioLoadingError
from lhotse.supervision import SupervisionSegment

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="download/voxpopuli_en",
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/voxpopuli_en_manifest",
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="en"
    )

    return parser

def get_meta_info(tsv_path, subset: str):
    def predicate(id: str):
        return id.endswith(subset)
    
    with gzip.open(tsv_path, mode="rt") as f:
        rows = [
            (r["event_id"], r["segment_no"], r["start"], r["end"])
            for r in csv.DictReader(f, delimiter="\t")
            if predicate(r["event_id"])
        ]
    return rows

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    subset = args.subset
    
    # create output directory
    os.makedirs(manifest_dir, exist_ok=True)
    
    audio_root = Path(f"{dataset_dir}/raw_audios")
    tsv_path = f"{dataset_dir}/unlabelled_data/unlabelled_v2.tsv.gz"
    manifest = get_meta_info(tsv_path, subset=subset)
    logging.info(f"Getting a total of {len(manifest)} items.")
    
    items = defaultdict(list)
    for event_id, seg_no, start, end in tqdm(manifest):
        lang, year = event_id.rsplit("_", 1)[1], event_id[:4]
        path = audio_root / lang / year / f"{event_id}.ogg"
        items[path.as_posix()].append((event_id, seg_no, float(start), float(end)))
    items = [(k, v) for k, v in items.items()]
    
    all_cuts = []
    num_cuts = 0
    
    import pdb; pdb.set_trace()
    for item in tqdm(items):
        audio_file, segments = item
        try:
            recording = Recording.from_file(audio_file)
        except AudioLoadingError:
            logging.info(f"Broken audio: {audio_file}, Skip this one.")
            continue
        for seg in segments:
            event_id, seg_num, start, end = seg
            duration = float(end) - float(start)
            cut_id = f"{event_id}_{seg_num}"
            cut = MonoCut(
                id=cut_id,
                start=float(start),
                duration=duration,
                channel=0,
                recording=recording,
            )
            sup = SupervisionSegment(
                id=cut_id,
                recording_id=cut.recording.id,
                start=0.0, # always zero
                channel=0,
                duration=duration,
                language="en",
            )
            cut.supervisions = [sup]
            all_cuts.append(cut)
            num_cuts += 1
            if num_cuts % 100 == 0:
                logging.info(f"Processed {num_cuts} cuts until now.")
    
    manifest_output_dir = f"{manifest_dir}/voxpopuli_cuts_{subset}.jsonl.gz"
    all_cuts = CutSet.from_cuts(all_cuts)
    logging.info(f"Storing the manifest to {manifest_output_dir}")
    all_cuts.to_jsonl(manifest_output_dir)
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()