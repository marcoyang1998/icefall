import os
from lhotse import load_manifest_lazy

def convert():
    old_folder = "data/vq_firered_zh_en_16_v2/"
    new_folder = "data_s3/vq_firered_zh_en_16_v2/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_source(c):
        source = c.recording.sources[0].source
        # source = source.replace("/fs-computility/INTERN6/shared/yangxiaoyu/", "s3://yangxiaoyu/")
        source = source.replace("/fs-computility/INTERN6/shared/yangxiaoyu/ASR_data/", "ASR:s3://ASR_20T/ASR_full_data/")
        # source = source.replace("/fs-computility/INTERN6/shared/yangxiaoyu/ASR_data/weread/weread-16k-res-0", "TTS:s3://tts/weread_16k_res_0")
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        return c
    
    # english data
    # datasets = ["peoplespeech", "common_voice_20200622"]
    # datasets += ["en_us_english", "en8848", "ljspeech", "tatoeba", "ted", "vctk", "voase", "voaSplider"]
    
    # chinese data
    datasets = ["accent", "aidatatang_200zh", "aishell3", "aishell2","baidu_en_cn","datatang1505"]
    datasets += ["dialog3k", "magicdata", "sensetime", "ximalaya", "acq", "cantonese", "cs_wav", "dialog"]
    datasets += ["MagicData_dialog","primewords_md_2018_set1","zhvoice","phone","speech_wav"]
    datasets += ["digital_library_202003", "ST-CMDS-20170001_1-OS", "20220309", "speech_annotations_2021"]
    datasets = ["speech_annotations_2021"]
    
    # datasets = [f"weread-16k-res-0{i}" for i in range(10)]
    
    for dataset in datasets:
        manifest_name = f"{dataset}_cuts.jsonl.gz"
        print(f"Start converting {dataset}.")
        target_manifest = new_folder + manifest_name
        if os.path.exists(target_manifest):
            print(f"Manifest already generated, skipping it")
            continue
        cuts = load_manifest_lazy(old_folder + manifest_name)
        
        cuts = cuts.map(change_source)
        audio = cuts[0].load_audio()
        print(audio.shape)
        
        print(f"Saving to {target_manifest}")
        cuts.to_jsonl(target_manifest)
    
    # dataset = "wenetspeech"
    # subsets = ["S","DEV","M","L","TEST_MEETING","TEST_NET"]
    # for subset in subsets:
    #     manifest_name = f"{dataset}_cuts_{subset}.jsonl.gz"
    #     cuts = load_manifest_lazy(old_folder + manifest_name)
        
    #     cuts = cuts.map(change_source)
    #     target_manifest = new_folder + manifest_name
    #     print(f"Saving to {target_manifest}")
    #     cuts.to_jsonl(target_manifest)

def convert_audioset():
    old_folder = f"data/vq_dasheng_large_layer_-1_normalize_0_cb_4/"
    new_folder = f"data_s3/vq_dasheng_large_layer_-1_normalize_0_cb_4/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_eval_source(c):
        source = c.recording.sources[0].source
        source = source.replace(
            "/mnt/workspace/xiaoyu/workspace/icefall_prompt_multi_task/egs/librispeech/ASR/download/audioset/eval/", 
            "brainllm:s3://yangxiaoyu/audioset/eval/wav_all/"
        )
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        return c
    
    def change_source(c):
        source = c.recording.sources[0].source
        source = source.replace(
            "/mnt/workspace/xiaoyu/workspace/icefall_prompt_multi_task/egs/librispeech/ASR/download/audioset/", 
            "brainllm:s3://yangxiaoyu/audioset/"
        )
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        return c
    
    subsets = ["balanced", "full", "eval"]
    for subset in subsets:
        manifest_name = f"audioset_cuts_{subset}.jsonl.gz"
        target_manifest = new_folder + manifest_name
        if os.path.exists(target_manifest):
            print(f"Manifest already generated, skipping it")
            continue
        cuts = load_manifest_lazy(old_folder + manifest_name)
        if subset == "eval":
            cuts = cuts.map(change_eval_source)
        else:
            cuts = cuts.map(change_source)
        
        print(f"Saving to {target_manifest}")
        cuts.to_jsonl(target_manifest)
        
    print(f"Finished")

def convert_audioset2():
    old_folder = f"data/fbank_as_ced_mAP50/"
    new_folder = f"data_s3/fbank_as_ced_mAP50/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_eval_source(c):
        source = c.recording.sources[0].source
        source = source.replace(
            "download/audioset/eval/wav_all/", 
            "brainllm:s3://yangxiaoyu/audioset/eval/wav_all/"
        )
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        return c
    
    def change_source(c):
        source = c.recording.sources[0].source
        source = source.replace(
            "download/audioset/", 
            "brainllm:s3://yangxiaoyu/audioset/"
        )
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        return c
    
    subsets = ["balanced", "full", "eval"]
    for subset in subsets:
        manifest_name = f"audioset_cuts_{subset}.jsonl.gz"
        target_manifest = new_folder + manifest_name
        if os.path.exists(target_manifest):
            print(f"Manifest already generated, skipping it")
            continue
        cuts = load_manifest_lazy(old_folder + manifest_name)
        if subset == "eval":
            cuts = cuts.map(change_eval_source)
        else:
            cuts = cuts.map(change_source)
        
        print(f"Saving to {target_manifest}")
        cuts.to_jsonl(target_manifest)
        
    print(f"Finished")


def convert_libriheavy():
    old_folder = f"data/vq_whisper_turbo_zh_en_16_v2_numpy/"
    new_folder = f"data_s3/vq_whisper_turbo_zh_en_16_v2_numpy/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_source(c):
        source = c.recording.sources[0].source
        source = source.replace("download/", "langchao2:s3://libriheavy/download/")
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        
        cb_source = c.codebook_indexes["path"]
        cb_source = cb_source.replace(
            "/cpfs02/user/housiyuan/xiaoyu/codebook_indexes/",
            "/mnt/petrelfs/zhangchen/xiaoyu/codebook_indexes/",
        )
        c.codebook_indexes["path"] = cb_source
        return c
    
    subsets = ["small", "medium", "large"]
    
    for subset in subsets:
        manifest_name = f"libriheavy_cuts_{subset}.jsonl.gz"
        print(f"Start converting {subset}.")
        target_manifest = new_folder + manifest_name
        if os.path.exists(target_manifest):
            print(f"Manifest already generated, skipping it")
            continue
        cuts = load_manifest_lazy(old_folder + manifest_name)
        
        cuts = cuts.map(change_source)
        
        print(f"Saving to {target_manifest}")
        cuts.to_jsonl(target_manifest)
        
def convert_libriheavy_split():
    old_folder = f"data/vq_hubert_large_layer_21_normalize_1_cb_16/"
    new_folder = f"data_s3/vq_hubert_large_layer_21_normalize_1_cb_16/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_source(c):
        source = c.recording.sources[0].source
        new_source = "brainllm:s3://yangxiaoyu/librilight_split/" + c.id + ".flac"
        c.recording.sources[0].source = new_source
        c.recording.sources[0].type = "url"
        
        cb_source = c.codebook_indexes["path"]
        cb_source = cb_source.replace(
            "/cpfs02/user/housiyuan/xiaoyu/codebook_indexes/",
            "brainllm:s3://yangxiaoyu/codebook_indexes/",
        )
        c.codebook_indexes["path"] = cb_source
        c.start = 0.0 # because we use short audio
        return c
    
    subsets = ["small", "medium", "large"]
    
    for subset in subsets:
        manifest_name = f"libriheavy_cuts_{subset}.jsonl.gz"
        print(f"Start converting {subset}.")
        target_manifest = new_folder + manifest_name
        if os.path.exists(target_manifest):
            print(f"Manifest already generated, skipping it")
            continue
        cuts = load_manifest_lazy(old_folder + manifest_name)
        
        cuts = cuts.map(change_source)
        
        print(f"Saving to {target_manifest}")
        cuts.to_jsonl(target_manifest)
        
def convert_gigaspeech():
    old_folder = f"data/vq_hubert_large_layer_21_normalize_1_cb_16/"
    new_folder = f"data_s3/vq_hubert_large_layer_21_normalize_1_cb_16/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_source(c):
        source = c.recording.sources[0].source
        new_source = source.replace(
            "/mnt/workspace/xiaoyu/workspace/icefall_multi_kd/egs/librispeech/ASR/",
            ""
        )
        new_source = new_source.replace(
            "download/gigaspeech/",
            "brainllm:s3://yangxiaoyu/gigaspeech/"
        )
        c.recording.sources[0].source = new_source
        c.recording.sources[0].type = "url"
        
        cb_source = c.codebook_indexes["path"]
        cb_source = cb_source.replace(
            "/cpfs02/user/housiyuan/xiaoyu/codebook_indexes/",
            "brainllm:s3://yangxiaoyu/codebook_indexes/",
        )
        c.codebook_indexes["path"] = cb_source
        c.start = 0.0 # because we use short audio
        return c
    
    subsets = ["dev", "xs", "s", "m", "l", "xl"]
    
    for subset in subsets:
        manifest_name = f"gigaspeech_cuts_{subset}.jsonl.gz"
        print(f"Start converting {subset}.")
        target_manifest = new_folder + manifest_name
        if os.path.exists(target_manifest):
            print(f"Manifest already generated, skipping it")
            continue

        cuts = load_manifest_lazy(old_folder + manifest_name).drop_features()
        
        cuts = cuts.map(change_source)
        
        print(f"Saving to {target_manifest}")
        cuts.to_jsonl(target_manifest)
        
def convert_librispeech():
    subset = "large"
    old_folder = f"data/vq_wavlm_large_layer_21_normalize_1_libri_cb_16/"
    new_folder = f"data_s3/vq_wavlm_large_layer_21_normalize_1_libri_cb_16/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_source(c):
        source = c.recording.sources[0].source
        source = source.replace(
            "/mnt/workspace/xiaoyu/workspace/icefall_prompt_multi_task/egs/librispeech/ASR/download/", 
            "brainllm:s3://yangxiaoyu/"
        )
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        
        if isinstance(c.codebook_indexes, dict):
            cb_source = c.codebook_indexes["path"]
            cb_source = cb_source.replace(
                "/cpfs02/user/housiyuan/xiaoyu/codebook_indexes/",
                "/mnt/petrelfs/zhangchen/xiaoyu/codebook_indexes/",
            )
            c.codebook_indexes["path"] = cb_source
        return c
    
    subsets = ["dev-clean", "dev-other", "train-all-shuf"]
    
    for subset in subsets:
        manifest_name = f"librispeech_cuts_{subset}.jsonl.gz"
        print(f"Start converting {subset}.")
        target_manifest = new_folder + manifest_name
        if os.path.exists(target_manifest):
            print(f"Manifest already generated, skipping it")
            continue
        cuts = load_manifest_lazy(old_folder + manifest_name)
        
        cuts = cuts.map(change_source)
        
        print(f"Saving to {target_manifest}")
        cuts.to_jsonl(target_manifest)

def convert_mls():
    old_folder = "data/vq_whisper_turbo_zh_en_16_v2/MLS_split/"
    new_folder = "data_s3/vq_whisper_turbo_zh_en_16_v2/MLS_split/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_source(c):
        source = c.recording.sources[0].source
        # source = source.replace("/fs-computility/INTERN6/shared/yangxiaoyu/", "s3://yangxiaoyu/")
        source = source.replace("/fs-computility/INTERN6/shared/yangxiaoyu/ASR_data/", "ASR:s3://ASR_20T/ASR_full_data/")
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        return c
    
    dataset = "MLS"
    
    for split in range(8):
        manifest_name = f"{dataset}_cuts.{split}.processed.jsonl.gz"
        print(f"Start converting {dataset} split {split}.")
        target_manifest = new_folder + manifest_name
        if os.path.exists(target_manifest):
            print(f"Manifest already generated, skipping it")
            continue
        
        cuts = load_manifest_lazy(old_folder + manifest_name)
        
        cuts = cuts.map(change_source)
        audio = cuts[0].load_audio()
        print(audio.shape)
        
        print(f"Saving to {target_manifest}")
        cuts.to_jsonl(target_manifest)

if __name__=="__main__":
    # convert_audioset()
    # convert_audioset2()
    # convert_libriheavy()
    # convert_libriheavy_split()
    # convert_librispeech()
    convert_gigaspeech()
        
        