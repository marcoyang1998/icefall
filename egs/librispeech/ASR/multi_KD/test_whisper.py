import whisper

model = whisper.load_model("large")
audio = "/star-xy/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
result = model.transcribe(audio)
print(result["text"])