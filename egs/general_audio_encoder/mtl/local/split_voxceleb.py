import logging

from lhotse import CutSet, load_manifest_lazy
from lhotse.supervision import SupervisionSegment

def chunk_audio_old(manifest, chunk_length=3.0, hop_length=0.0):
    cuts = load_manifest_lazy(manifest)
    chunked_cuts = cuts.cut_into_windows(
        duration=chunk_length,
        hop=hop_length,
        keep_excessive_supervisions=True,
    )
    return chunked_cuts

def chunk_audio(manifest, chunk_length=3.0, hop_length=0.0):
    cuts = load_manifest_lazy(manifest)
    new_cuts = []
    num_cuts = 0
    
    for cut in cuts:
        chunked_cuts = cut.cut_into_windows(chunk_length, hop_length)
        if cut.duration > 4.0:
            for cc in chunked_cuts:
                sup = SupervisionSegment(
                    id=cc.id,
                    recording_id=cut.recording.id,
                    start=0.0, # always zero
                    channel=0,
                    duration=cc.duration,
                    speaker=cut.supervisions[0].speaker,
                )
                cc.supervisions = [sup]
                num_cuts += 1
                if num_cuts % 1000 == 0:
                    logging.info(f"Processed {num_cuts} cuts until now.")
                new_cuts.append(cc)
        else:
            new_cuts.append(cut)
            num_cuts += 1
            if num_cuts % 1000 == 0:
                logging.info(f"Processed {num_cuts} cuts until now.")
    
    logging.info(f"After chunking, a total of {len(new_cuts)} valid samples.")
    new_cuts = CutSet.from_cuts(new_cuts)
    return new_cuts

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    
    manifests = [
        "data/fbank_voxceleb/vox1_cuts_dev.jsonl.gz",
        "data/fbank_voxceleb/vox2_cuts_dev.jsonl.gz",
    ]
    chunk_len = 3
    for m in manifests:
        chunked = chunk_audio(m, chunk_length=chunk_len, hop_length=0.0)
        out_path = m.replace(".jsonl.gz", f"_{chunk_len}s.jsonl.gz")
        chunked.to_jsonl(out_path)
        print(f"Wrote chunked cuts to {out_path}")