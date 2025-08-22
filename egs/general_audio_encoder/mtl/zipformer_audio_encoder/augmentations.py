import random

from lhotse.cut import CutSet, MonoCut, Cut
from lhotse.cut.set import mix

def mix_with_offset(
    reference_cut: Cut,
    mixed_in_cut: Cut,
    snr: float = 10.0,
    drop_mixed_in_supervision: bool = True
):
    if drop_mixed_in_supervision:
        mixed_in_cut = mixed_in_cut.drop_supervisions()
    ref_duration = reference_cut.duration
    mixed_in_duration = mixed_in_cut.duration
    
    mix_duration = random.uniform(0, ref_duration / 2)
    
    # randomly truncate the mixed_in_cut to mix_duration if longer
    if mixed_in_duration > mix_duration:
        diff = mixed_in_duration - mix_duration
        truncate_start = random.uniform(0, diff)
        mixed_in_cut = mixed_in_cut.truncate(offset=truncate_start, duration=mix_duration)
        
    actual_mix_duration = min(mixed_in_cut.duration, mix_duration)
    offset = random.uniform(0, ref_duration - actual_mix_duration - 0.05) # a tolerance of 0.05 for safety
    mixed_cut = mix(
        reference_cut=reference_cut,
        mixed_in_cut=mixed_in_cut,
        offset=offset,
        snr=snr,
        preserve_id="left",
    )
    
    return mixed_cut

class BatchMixing:
    def __init__(
        self,
        min_snr: float = 10,
        max_snr: float = 20,
        p: float = 0.5,
        drop_mixed_in_supervision: bool = True,
    ):
        """perform in-batch mixing with the cuts from the same batch

        Args:
            min_snr (float): minimum mix SNR
            max_snr (float): maximum mix SNR
            p (float, optional): The probability of perform mixing to a cut. Defaults to 0.5.
            drop_mixed_in_supervision (bool, optional): Remove the supervisions in the mixed_in_cut. Defaults to True.
        """
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p
        self.drop_mixed_in_supervision = drop_mixed_in_supervision
        
    def __call__(self, reference_cuts: CutSet) -> CutSet:
        noise_cuts = reference_cuts # in batch mixing
        results = []
        for cut in reference_cuts:
            if random.random() < self.p and noise_cuts is not None:
                snr = random.uniform(self.min_snr, self.max_snr)
                mixed_in_cut = noise_cuts.sample(n_cuts=1)
                while mixed_in_cut.id == cut.id:
                    mixed_in_cut = noise_cuts.sample(n_cuts=1)
                mixed_cut = mix_with_offset(cut, mixed_in_cut, snr=snr)
                results.append(mixed_cut)
            else:
                results.append(cut)
        return CutSet.from_cuts(results)
    
def _test_mix():
    from lhotse import load_manifest_lazy
    manifest = "data/fbank/librispeech_cuts_dev-other.jsonl.gz"
    cuts = load_manifest_lazy(manifest).drop_features().subset(first=10)
    
    transform = BatchMixing()
    mixed_cuts = transform(cuts)
    print(mixed_cuts)
    
if __name__=="__main__":
    _test_mix()