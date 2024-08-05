#!/usr/bin/env bash
files=$(find downloads/fma/data/fma_large -name "*.mp3")

for file in $files; do
    ffmpeg -i $file -acodec pcm_s16le -ar 16000 ${file%.mp3}.wav -loglevel quiet
done