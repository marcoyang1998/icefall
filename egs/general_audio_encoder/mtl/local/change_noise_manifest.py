from lhotse import load_manifest_lazy

manifests = [
    "audioset_non_human.jsonl.gz",
    "musan_non_speech.jsonl.gz",
    "musan_speech.jsonl.gz",
    "noise_non_speech_musan_audioset.jsonl.gz",
]


for m in manifests:
    manifest = f"data/musan_old/{m}"
    output_manifest = f"data/musan/{m}"

    cuts = load_manifest_lazy(manifest)

    def change_audio_root(c):
        source = c.recording.sources[0].source
        
        source = source.replace(
            "/mnt/workspace/xiaoyu/workspace/icefall_prompt_multi_task/egs/librispeech/ASR/download/",
            "download/",
        )
        c.recording.sources[0].source = source
        if c.has_custom("beats_embedding"):
            del c.beats_embedding
        return c

    cuts = cuts.map(change_audio_root)

    cuts.to_jsonl(output_manifest)
