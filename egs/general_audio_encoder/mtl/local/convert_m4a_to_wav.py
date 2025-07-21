import os
import subprocess

def convert_m4a_to_wav(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".m4a"):
                input_path = os.path.join(dirpath, filename)

                # 构建对应的输出路径，保留目录结构
                rel_path = os.path.relpath(input_path, root_dir)
                rel_path = os.path.splitext(rel_path)[0] + ".wav"
                output_path = os.path.join(output_dir, rel_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # ffmpeg 命令：转成 16kHz, mono wav
                command = [
                    "ffmpeg",
                    "-i", input_path,
                    "-ac", "1",       # 单声道
                    "-ar", "16000",   # 采样率 16000 Hz
                    "-y",             # 自动覆盖
                    output_path
                ]

                print(f"🔄 Converting: {input_path} → {output_path}")
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✅ 所有 m4a 文件已转换完毕。")

if __name__ == "__main__":
    input_folder = "download/MEAD"     # 替换为你的输入路径
    output_folder = "download/MEAD" # 替换为输出路径
    convert_m4a_to_wav(input_folder, output_folder)