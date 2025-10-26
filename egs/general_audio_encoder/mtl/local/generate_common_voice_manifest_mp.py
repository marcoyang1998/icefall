import argparse
import csv
import glob
import logging
import os
from tqdm import tqdm
import pandas as pd
import torch
from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.audio.utils import AudioLoadingError
from lhotse.supervision import SupervisionSegment
from concurrent.futures import ProcessPoolExecutor, as_completed

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default="download/cv-corpus-4.0")
    parser.add_argument("--manifest-dir", type=str, default="data/cv_manifest")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--num-workers", type=int, default=8, help="number of parallel workers")
    return parser


def parse_tsv(tsv_file: str):
    df = pd.read_csv(tsv_file, sep="\t")
    df = df[["path", "sentence", "gender"]]
    return df


def generate_audio_mapping(audio_root: str):
    audio_files = glob.glob(f"{audio_root}/*/*.wav")
    def get_audio_id(s: str):
        filename = os.path.basename(s)
        return os.path.splitext(filename)[0]
    audio_mapping = {get_audio_id(audio_file): audio_file for audio_file in audio_files}
    return audio_mapping


def process_row(row, audio_mapping, language):
    """单个样本的处理逻辑，供多进程调用"""
    path = row["path"]
    sample_id = path.replace(".mp3", "").replace(".wav", "")
    if sample_id not in audio_mapping:
        return None  # 跳过未找到的文件

    audio_file = audio_mapping[sample_id]
    if audio_file.endswith(".mp3"):
        audio_file = audio_file.replace(".mp3", ".wav")

    try:
        recording = Recording.from_file(audio_file)
    except AudioLoadingError:
        return None

    cut = MonoCut(
        id=sample_id,
        start=0,
        duration=recording.duration,
        channel=0,
        recording=recording,
    )

    supervision = SupervisionSegment(
        id=sample_id,
        recording_id=cut.recording.id,
        start=0.0,
        channel=0,
        duration=cut.duration,
        text=row["sentence"],
        language=language,
        gender=row["gender"],
    )
    cut.supervisions = [supervision]
    return cut


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    subset = args.subset
    language = args.language
    num_workers = args.num_workers

    os.makedirs(manifest_dir, exist_ok=True)

    tsv_file = f"{dataset_dir}/transcript/{language}/{subset}.tsv"
    meta_info = parse_tsv(tsv_file)
    audio_root = f"{dataset_dir}/audio/{language}/{subset}"
    audio_mapping = generate_audio_mapping(audio_root)

    logging.info(f"Processing {len(meta_info)} entries with {num_workers} workers...")

    cuts = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_row, row, audio_mapping, language)
            for _, row in meta_info.iterrows()
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            if result is not None:
                cuts.append(result)

    logging.info(f"After filtering, a total of {len(cuts)} valid samples.")
    cuts = CutSet.from_cuts(cuts).resample(16000)

    manifest_output = os.path.join(
        manifest_dir, f"commonvoice_{language}_cuts_{subset}.jsonl.gz"
    )
    logging.info(f"Storing the manifest to {manifest_output}")
    cuts.to_jsonl(manifest_output)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    main()