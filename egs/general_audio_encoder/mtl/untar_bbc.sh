#!/bin/bash

# 设定存放 .tar 文件的目录（当前目录）
TAR_DIR="./download/BBCSoundEffects/test"  # 可以改为你自己的路径，例如 /data/my_tars

# 输出目录（解压后文件放在哪里）
OUT_DIR="./download/BBCSoundEffects/test"

# 解压 0.tar 到 28.tar
for i in $(seq 0 3); do
    tar_file="${TAR_DIR}/${i}.tar"
    if [ -f "$tar_file" ]; then
        echo "🟢 正在解压: $tar_file"
        tar -xf "$tar_file" -C "$OUT_DIR"
    else
        echo "⚠️  未找到: $tar_file，跳过"
    fi
done

echo "✅ 所有文件处理完毕，输出目录：$OUT_DIR"