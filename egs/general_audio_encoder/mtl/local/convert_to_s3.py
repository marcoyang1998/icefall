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

def convert_libriheavy():
    subset = "large"
    old_folder = f"data/vq_whisper_turbo_zh_en_16_v2/libriheavy_{subset}_split/"
    new_folder = f"data_s3/vq_whisper_turbo_zh_en_16_v2/libriheavy_{subset}_split/"
    
    os.makedirs(new_folder, exist_ok=True)
    
    def change_source(c):
        source = c.recording.sources[0].source
        # source = source.replace("/fs-computility/INTERN6/shared/yangxiaoyu/", "s3://yangxiaoyu/")
        # source = source.replace("/fs-computility/INTERN6/shared/yangxiaoyu/ASR_data/", "ASR:s3://ASR_20T/ASR_full_data/")
        source = source.replace("download/", "langchao2:s3://libriheavy/download/")
        c.recording.sources[0].source = source
        c.recording.sources[0].type = "url"
        return c
    
    # datasets = ["peoplespeech", "common_voice_20200622"]
    # datasets += ["en_us_english", "en8848", "ljspeech", "tatoeba", "ted", "vctk", "voase", "voaSplider"]
    
    datasets = [str(i) for i in range(25)]
    
    for dataset in datasets:
        manifest_name = f"libriheavy_cuts_{subset}.{dataset}.processed.jsonl.gz"
        print(f"Start converting split {dataset}.")
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
    convert()
        
        