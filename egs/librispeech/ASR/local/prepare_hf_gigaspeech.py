import argparse
import pandas as pd
import glob
import os
from tqdm import tqdm
import logging

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="download/gigaspeech"
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="dev",
    )
    
    parser.add_argument(
        "--manifest-folder",
        type=str,
        default="data/manifests"
    )
    
    return parser.parse_args()

def parse_metadata(csv_file):
    df = pd.read_csv(csv_file)
    result_dict = df.set_index('sid')['text_tn'].to_dict()
    return result_dict

def main():
    args = get_parser()
    logging.info(vars(args))
    
    subset = args.subset
    manifest_dir = args.manifest_folder
    
    output_manifest = f"{args.manifest_folder}/gigaspeech_cuts_{subset}.jsonl.gz"
    if os.path.exists(output_manifest):
        logging.info(f"{subset} already processed! Manifest is here: {output_manifest}. Skip it.")
        return
    
    if subset in ["dev", "test", "xs"]:
        dataset_dir = f"{args.dataset_dir}/data/audio/{subset}_files"
        metadata_dir = f"{args.dataset_dir}/data/metadata/{subset}_metadata"
        manifest_dir = f"{manifest_dir}/gigaspeech_{subset}_splits"
    else:
        dataset_dir = f"{args.dataset_dir}/data/audio/{subset}_files_additional"
        metadata_dir = f"{args.dataset_dir}/data/metadata/{subset}_metadata_additional"
        manifest_dir = f"{manifest_dir}/gigaspeech_{subset}_splits"
    
    if not os.path.exists(manifest_dir):
        os.mkdir(manifest_dir)
    
    csv_files = sorted(glob.glob(f"{metadata_dir}/*.csv"))
    num_chunks = len(csv_files)
    logging.info(f"Obtaining {len(csv_files)} chunks of {subset}.")
    
    for i in tqdm(range(num_chunks)):
        logging.info(f"Start processing {csv_files[i]}")
        cuts_file = f"{manifest_dir}/gigaspeech_cuts_{subset}_chunks_{str(i).zfill(4)}.jsonl.gz"
        if os.path.exists(cuts_file):
            logging.info(f"Chunk {i} already preprocessed at {cuts_file}. Skip it.")
            continue
        
        wav_folder = f"{dataset_dir}/{subset}_chunks_{str(i).zfill(4)}"
        recordings = []
        supervisions = []
        result_dict= parse_metadata(csv_file=csv_files[i])
        for id, text in result_dict.items():
            wav_file = f"{wav_folder}/{id}.wav"
            try:
                audio = Recording.from_file(wav_file)
                recordings.append(audio)
                supervision = SupervisionSegment(
                    id=id,
                    recording_id=audio.id,
                    text=text,
                    start=0.0,
                    channel=0,
                    duration=audio.duration,
                )
                supervisions.append(supervision)
            except:
                logging.info(f"Skipping {wav_file}. The audio is corrupted")
            
        recordings = RecordingSet.from_recordings(recordings)
        recordings.to_jsonl(f"{manifest_dir}/gigaspeech_recordings_{subset}_chunks_{str(i).zfill(4)}.jsonl.gz")
        supervisions = SupervisionSet.from_segments(supervisions)
        supervisions.to_jsonl(f"{manifest_dir}/gigaspeech_supervisions_{subset}_chunks_{str(i).zfill(4)}.jsonl.gz")
        cuts = CutSet.from_manifests(
            recordings=recordings,
            supervisions=supervisions,
        )
        cuts.to_jsonl(cuts_file)
        logging.info(f"Saved to {cuts_file}.")
        logging.info(f"Finish processing chunk {i}")
    
    if num_chunks > 1:
        chunk_cuts = f"{manifest_dir}/gigaspeech_cuts_{subset}_chunks_*.jsonl.gz"
        combine_cmd = f"lhotse combine {chunk_cuts} {output_manifest}"
        os.system(combine_cmd)
    else:
        chunk_cuts = f"{manifest_dir}/gigaspeech_cuts_{subset}_chunks_0000.jsonl.gz"
        rename_cmd = f"cp {chunk_cuts} {output_manifest}"
        os.system(rename_cmd)
    logging.info(f"Saved to {output_manifest}")
    logging.info(f"Finished for {subset}")

    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    main()