import time
from kokoro_onnx import Kokoro

model_path = "kokoro-v1.0.onnx"
voices_path = "voices-v1.0.bin"

kokoro = Kokoro(model_path, voices_path)
with open("previews/demo-zh.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 1. With trim=True
t0 = time.time()
audio, sr = kokoro.create(text, voice="zf_xiaobei", lang="en-us", trim=True)
print(f"With trim=True took: {time.time() - t0:.4f} seconds")

# 2. With trim=False
t0 = time.time()
audio, sr = kokoro.create(text, voice="zf_xiaobei", lang="en-us", trim=False)
print(f"With trim=False took: {time.time() - t0:.4f} seconds")
