import os
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# 配置参数
CSV_PATH = "download/vggsound.csv"
OUTPUT_DIR = Path("download/vggsound_10s")
NUM_WORKERS = 16
DURATION = 10
SAMPLE_RATE = 16000
CHANNELS = 1

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_row(ytid, start):
    try:
        output_wav = OUTPUT_DIR / f"{ytid}_{start:.2f}_{start+DURATION:.2f}.wav"
        if output_wav.exists():
            return

        audio_path = f"{ytid}.m4a"

        # 下载音频（如果不存在）
        if not os.path.exists(audio_path):
            subprocess.run([
                "yt-dlp",
                "-f", "bestaudio",
                "--quiet",
                "--no-warnings",
                "-o", audio_path,
                f"https://www.youtube.com/watch?v={ytid}"
            ], check=True)

        # 精确裁剪音频
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(DURATION),
            "-i", audio_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-loglevel", "error",
            str(output_wav)
        ], check=True)

        # 删除原始音频
        if os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        print(f"❌ Error processing {ytid} {start}s: {e}")

def main():
    df = pd.read_csv(CSV_PATH, comment='#', header=None,
                 names=['ytid', 'start', 'label', 'split'])
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_row, row["ytid"], float(row["start"]))
            for _, row in df.iterrows()
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass

if __name__ == "__main__":
    main()