import re

# 输入和输出文件名
input_filename = "dataset_cuts.txt"  # 替换为你的输入文件
output_filename = "data/stats/stats_len.txt"

# 处理数据
with open(input_filename, "r", encoding="utf-8") as infile, open(output_filename, "w", encoding="utf-8") as outfile:
    for line in infile:
        match = re.match(r"([^:]+):\│ Cuts count:\s+\│\s+(\d+)", line)
        if match:
            dataset_name = match.group(1).replace("_cuts_stats.txt", "")  # 去掉后缀
            file_count = match.group(2)  # 获取文件数
            
            # 写入新的文件
            outfile.write(f"{dataset_name}\t{file_count}\n")

print(f"处理完成，结果已保存到 {output_filename}")