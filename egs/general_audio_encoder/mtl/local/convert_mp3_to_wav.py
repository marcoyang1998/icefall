import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

def convert_file(input_path, output_path):
    # 如果输出文件已存在且大小大于0，跳过
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"⏭️ 已存在，跳过: {output_path}")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-y",
        output_path
    ]

    print(f"🔄 Converting: {input_path} → {output_path}")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def gather_files(root_dir, output_dir):
    tasks = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mp3"):
                input_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(input_path, root_dir)
                rel_path = os.path.splitext(rel_path)[0] + ".wav"
                output_path = os.path.join(output_dir, rel_path)
                tasks.append((input_path, output_path))
    return tasks

def convert_mp3_to_wav_multithreaded(root_dir, output_dir, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    tasks = gather_files(root_dir, output_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_file, inp, outp) for inp, outp in tasks]
        for future in as_completed(futures):
            future.result()  # Raise exceptions if any

    print("✅ 所有 mp3 文件已转换完毕。")

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        input_folder = args[0]
        output_folder = input_folder
    elif len(args) == 2:
        input_folder, output_folder = args
    else:
        raise NotImplementedError(f"Unexpected number of args!")
    # output_folder = "download/common_voice_17_0/audio/en/other"  # 可设置成不同路径
    convert_mp3_to_wav_multithreaded(input_folder, output_folder, max_workers=8)