#!/bin/bash
cd download/common_voice_17_0

find . -type f -name "*.tar" | grep -i "train" | while read -r file; do
    echo "Downloading with git lfs: $file"
    git lfs pull --include="$file"
done