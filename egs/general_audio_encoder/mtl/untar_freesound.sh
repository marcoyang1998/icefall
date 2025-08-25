#!/bin/bash

# 检查是否提供了目标路径
if [ -z "$1" ]; then
  echo "用法: $0 目标路径"
  exit 1
fi

TARGET_DIR="$1"

# 检查目标路径是否存在
if [ ! -d "$TARGET_DIR" ]; then
  echo "错误：路径 '$TARGET_DIR' 不存在或不是目录"
  exit 1
fi

# 查找并解压所有.tar文件
cd $TARGET_DIR
files=$(find . -name "*.tar")
for file in $files; do
    echo "Untarring ${file}"
    tar -xf $file
done

echo "全部解压完成。"