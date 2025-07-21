import json
import logging
import os
import glob
from pathlib import Path
from ast import literal_eval

import torch
from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment


def parse_supervisions(txt_file: str):
    supervisions = []
    with open(txt_file, "r") as f:
        for i, line in enumerate(f):
            if i <=2:
                continue
            if line.strip() == "":
                continue
            start, end, speaker, text = line.strip().split(" ", 3)
            start = float(start)
            end = float(end)
            speaker = speaker.replace(":", "")
            supervisions.append(
                [start, end, speaker, text]
            )
    return supervisions

def prepare_fisher():
    part = 2
    logging.info(f"Processing part {part}")
    meta_file_folder = f"download/Fisher/fe_03_p{part}_tran/data/trans"
    audio_file_folder = Path(f"download/Fisher/fisher_part{part}_wav_16k")
    all_recordings = audio_file_folder.rglob("*.wav")
    
    cuts = []
    num_cuts = 0
    for audio_file in all_recordings:
        audio_file = str(audio_file)
        recording = Recording.from_file(audio_file) # shared
        
        folder_id = audio_file.split("/")[-2]
        audio_name = audio_file.split("/")[-1].replace(".wav", "")
        meta_file = meta_file_folder + f"/{folder_id}/{audio_name}.txt"
        raw_supervisions = parse_supervisions(meta_file)

        for i, info in enumerate(raw_supervisions):
            start, end, speaker, text = info
            cut_id = audio_name + f"-{i}"
            
            cut = MonoCut(
                id=cut_id,
                start=start,
                duration=end - start,
                channel=0, # we only use the first channel
                recording=recording,
            )
            
            sup = SupervisionSegment(
                id=cut_id,
                recording_id=cut.recording.id,
                start=0.0,
                duration=cut.duration,
                speaker=speaker,
                text=text,
                channel=0
            )
            
            cut.supervisions = [sup]
            cuts.append(cut)
            num_cuts += 1
            if num_cuts % 100 == 0 and num_cuts:
                logging.info(f"Processed {num_cuts} cuts until now.")
            
    import pdb; pdb.set_trace()
    cuts = CutSet.from_cuts(cuts)
    
    output_manifest = f"data/fisher_manifest/fisher_cuts_part{part}.jsonl.gz"
    logging.info(f"Saving the manfiest to {output_manifest}")
    cuts.to_jsonl(output_manifest)
    logging.info("Done")

        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    prepare_fisher()