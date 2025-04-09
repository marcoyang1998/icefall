#!/usr/bin/env bash

# 配置参数
BUCKET_NAME=yangxiaoyu  # 你的 S3 存储桶
MOUNT_POINT="/mnt/cache/share_data/zhangchen/s3mount" # 你的挂载路径

# 检查是否已经挂载
if findmnt -r "$MOUNT_POINT" > /dev/null 2>&1; then
    echo "Goofys is already mounted on $MOUNT_POINT. Unmounting..."
    fusermount -u "$MOUNT_POINT"
    sleep 2  # 给系统一些时间来清理挂载
fi

# 重新挂载
echo "Mounting $BUCKET_NAME to $MOUNT_POINT using Goofys..."
/mnt/petrelfs/share_data/housiyuan/softwares/goofys --endpoint=http://p-ceph-norm-inside.pjlab.org.cn $BUCKET_NAME $MOUNT_POINT

# 检查挂载是否成功
if findmnt -r "$MOUNT_POINT" > /dev/null 2>&1; then
    echo "Mount successful!"
else
    echo "Mount failed!"
    exit 1
fi