

for subset in dev test xs s m l xl; do
    python local/generate_gigaspeech_manifest.py \
        --manifest-dir data/gigaspeech_manifest \
        --subset $subset \
        --dataset-dir download/gigaspeech
done