import argparse
import os
import glob
import re
from collections import defaultdict

def parse_log_files(log_dir):
    # 匹配所有 log-decode-epoch-* 文件
    log_files = glob.glob(os.path.join(log_dir, "log-decode-epoch-*"))

    wer_dict = defaultdict(dict)  # {model_id: {"clean": val, "other": val}}

    for file_path in log_files:
        with open(file_path, 'r') as f:
            for line in f:
                if "best for test-clean" in line or "best for test-other" in line:
                    # 提取 WER 和标签
                    match = re.search(r'([\d.]+)\s+best for (test-clean|test-other)', line)
                    if match:
                        wer = float(match.group(1))
                        label = match.group(2)
                        # 使用文件名（去掉路径）作为模型 ID
                        model_id = os.path.basename(file_path)
                        wer_dict[model_id][label] = wer

    return wer_dict

def main():
    parser = argparse.ArgumentParser(description="Analyze WER from decode logs.")
    parser.add_argument("log_dir", help="Directory containing log-decode-epoch-* files")
    args = parser.parse_args()

    wer_dict = parse_log_files(args.log_dir)

    result = []
    for model_id, values in wer_dict.items():
        if "test-clean" in values and "test-other" in values:
            clean = values["test-clean"]
            other = values["test-other"]
            avg = (clean + other) / 2
            result.append((avg, model_id, clean, other))

    # 按平均 WER 排序
    result.sort()

    # 输出
    print(f"{'Model':<50} {'Clean':>8} {'Other':>8} {'Avg':>8}")
    print("-" * 80)
    for avg, model_id, clean, other in result:
        print(f"{model_id:<50} {clean:>8.2f} {other:>8.2f} {avg:>8.2f}")

if __name__ == "__main__":
    main()