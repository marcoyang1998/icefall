import os
import subprocess
from pathlib import Path

# 原始 sph 根目录
input_dir = Path("./download/Fisher/fisher_part2")
# 输出 wav 根目录
output_dir = Path("./download/Fisher/fisher_part2_wav_16k")

# 创建输出目录
output_dir.mkdir(parents=True, exist_ok=True)

# 遍历所有 .sph 文件
for sph_path in input_dir.rglob("*.sph"):
    rel_path = sph_path.relative_to(input_dir)
    wav_path = output_dir / rel_path.with_suffix(".wav")

    # 创建输出子目录
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"⏳ Converting {sph_path} -> {wav_path}")

    # 1. 用 sph2pipe 解码成 wav
    try:
        # 临时路径（未重采样）
        tmp_wav = wav_path.with_suffix(".tmp.wav")

        with open(tmp_wav, "wb") as f:
            subprocess.run(["sph2pipe", "-f", "wav", str(sph_path)], stdout=f, check=True)

        # 2. 用 ffmpeg 重采样到 16kHz 单声道
        subprocess.run([
            "ffmpeg", "-y", "-i", str(tmp_wav),
            "-ar", "16000", "-ac", "1",
            str(wav_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 删除中间文件
        os.remove(tmp_wav)

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {sph_path} ({e})")