import os
import concurrent.futures
import subprocess

def convert_opus_to_wav(input_file):
    """使用 ffmpeg 将 opus 转换为 16kHz 16-bit PCM WAV"""
    output_file = os.path.splitext(input_file)[0] + ".wav"
    temp_output_file = output_file + ".tmp"
    
    if os.path.exists(output_file):
        print(f"已存在，跳过: {output_file}")
        return
    
    command = [
        "ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", temp_output_file
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        os.rename(temp_output_file, output_file)
        print(f"转换成功: {input_file} -> {output_file}")
    except subprocess.CalledProcessError:
        print(f"转换失败: {input_file}")
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)

def find_all_opus_files(folder):
    """递归查找所有 .opus 文件"""
    opus_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".opus"):
                opus_files.append(os.path.join(root, file))
    return opus_files

def batch_convert_opus_to_wav(folder, max_workers=4):
    """多线程批量转换所有 opus 文件，包括子文件夹"""
    opus_files = find_all_opus_files(folder)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(convert_opus_to_wav, opus_files)

if __name__ == "__main__":
    folder = "/fs-computility/INTERN6/shared/yangxiaoyu/WenetSpeech/wenetspeech/audio/test_net"  # OPUS 和 WAV 文件存放目录
    batch_convert_opus_to_wav(folder, max_workers=8)