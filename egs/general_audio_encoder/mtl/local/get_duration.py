import re

# 读取输入数据文件（假设它是一个文本文件）
input_filename = "dataset_durations.txt"  # 请替换为你的文件名
output_filename = "data/stats/stats_duration.txt"

# 解析并转换时间的函数
def parse_duration(duration_str):
    """ 将时长 'HH:MM:SS' 转换为小时（四舍五入） """
    h, m, s = map(int, duration_str.split(":"))
    total_hours = h + m / 60 + s / 3600
    return round(total_hours)  # 四舍五入到整数小时

# 处理数据
with open(input_filename, "r", encoding="utf-8") as infile, open(output_filename, "w", encoding="utf-8") as outfile:
    for line in infile:
        match = re.match(r"([^:]+):\│ Total speech duration\s+\│\s+([\d:]+)", line)
        if match:
            dataset_name = match.group(1).replace("_cuts_stats.txt", "")  # 去掉后缀
            duration_str = match.group(2)  # 获取时长
            total_hours = parse_duration(duration_str)  # 转换成小时
            
            # 写入新的文件
            outfile.write(f"{dataset_name}\t{total_hours}\n")

print(f"处理完成，结果已保存到 {output_filename}")