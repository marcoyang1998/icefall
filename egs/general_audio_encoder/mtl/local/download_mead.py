import os
import subprocess
import tarfile

def download_from_gdrive(speaker_id, gdrive_url, output_dir="/cpfs02/shared/speechllm/xiaoyu/MEAD"):
    os.makedirs(output_dir, exist_ok=True)
    speaker_dir = os.path.join(output_dir, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)

    # 提取文件ID
    if "id=" in gdrive_url:
        file_id = gdrive_url.split("id=")[1].split("&")[0]
    else:
        file_id = gdrive_url.split("/d/")[1].split("/")[0]

    # 下载路径
    tar_path = os.path.join(speaker_dir, "audio.tar")
    folder_path = os.path.join(speaker_dir, "audio")
    if os.path.exists(folder_path):
        print(f"Skipping {speaker_id}, already downloaded!")
        return

    # 下载 audio.tar 文件
    print(f"⬇️ Downloading for speaker {speaker_id}...")
    subprocess.run(["gdown", "--id", file_id, "-O", tar_path], check=True)

    # 解压 audio.tar
    print(f"📦 Extracting audio.tar into {speaker_dir}...")
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=speaker_dir)
        os.remove(tar_path)
        print(f"✅ Done for {speaker_id}")
    except Exception as e:
        print(f"❌ Error extracting {tar_path}: {e}")

def process_txt_file(txt_path):
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "http" not in line:
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"⚠️ Skipping invalid line: {line}")
                continue
            speaker_id, gdrive_url = parts
            try:
                download_from_gdrive(speaker_id, gdrive_url)
            except:
                print(f"Failed for {speaker_id}.")

if __name__ == "__main__":
    txt_file = "files.txt"  # 修改为你的文件名
    process_txt_file(txt_file)