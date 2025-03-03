import os

import torch
from lhotse import load_manifest_lazy

from icefall.utils import get_executor, str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def save_audio():
    num_jobs = min(45, os.cpu_count())
    print(f"Num jobs: {num_jobs}")

    for subset in ["DEV", "S", "L", "M", "TEST_MEETING", "TEST_NET"]:
        output_folder = f"/fs-computility/INTERN6/shared/yangxiaoyu/wenetspeech_trimmed/{subset}"
        output_manifest = f"data/fbank_wenetspeech_wav_trimmed/wenetspeech_cuts_{subset}.jsonl.gz"
        
        if os.path.exists(output_manifest):
            print(f"Skipping {subset}, already created!")
            continue
        
        cuts = load_manifest_lazy(f"data/fbank_wenetspeech_wav/wenetspeech_cuts_{subset}.jsonl.gz")

        with get_executor() as ex:
            cuts = cuts.save_audios(
                storage_path=output_folder,
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
            )

        cuts = cuts.to_jsonl(output_manifest)
    
if __name__=="__main__":
    save_audio()