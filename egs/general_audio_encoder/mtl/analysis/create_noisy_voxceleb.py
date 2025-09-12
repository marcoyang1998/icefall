from lhotse import load_manifest_lazy, load_manifest
from lhotse.dataset import CutMix

def create_noisy_voxceleb():

    cuts = load_manifest_lazy("data/fbank_voxceleb/vox1_cuts_test.jsonl.gz")

    noise_cuts = load_manifest("data/musan/musan_non_speech.jsonl.gz")

    t = CutMix(
        noise_cuts, p=1.0, snr=5, preserve_id=True, pad_to_longest=False
    )

    noisy_cuts = t(cuts)

    import pdb; pdb.set_trace()

    noisy_cuts.to_jsonl("data/manifests/vox1_cuts_test_noisy.jsonl.gz")
    
def create_noisy_librispeech():
    
    cuts = load_manifest_lazy("data/fbank_librispeech/librispeech_cuts_dev-clean.jsonl.gz")

    noise_cuts = load_manifest("data/musan/musan_non_speech.jsonl.gz")

    t = CutMix(
        noise_cuts, p=1.0, snr=5, preserve_id=True, pad_to_longest=False
    )

    noisy_cuts = t(cuts)

    import pdb; pdb.set_trace()

    noisy_cuts.to_jsonl("data/manifests/librispeech_dev_clean_noisy.jsonl.gz")
    
if __name__=="__main__":
    # create_noisy_voxceleb()
    create_noisy_librispeech()