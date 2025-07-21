import os
import subprocess
import sys

def convert_mp4_to_wav(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(input_dir, output_filename)

            command = [
                "ffmpeg",
                "-i", input_path,
                "-vn",         # 去除视频
                "-ac", "1",    # 单通道
                "-ar", "16000",# 采样率 16kHz
                "-y",          # 覆盖已有文件（如不想覆盖可以去掉）
                output_path
            ]

            print(f"🔄 正在转换: {filename} → {output_filename}")
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("✅ 所有 mp4 文件已转换为 16kHz 单通道 wav 文件")

if __name__ == "__main__":
    input_dir=str(sys.argv[1])
    convert_mp4_to_wav(input_dir)