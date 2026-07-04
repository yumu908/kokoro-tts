import time

t_start = time.time()

# 1. Measure Import time
import os
import onnxruntime as ort
from kokoro_onnx import Kokoro
t_imports = time.time() - t_start
print(f"1. Imports took: {t_imports:.4f} seconds")

# 2. Measure Initialization time
t0 = time.time()
model_path = "kokoro-v1.0.onnx"
voices_path = "voices-v1.0.bin"
kokoro = Kokoro(model_path, voices_path)
t_init = time.time() - t0
print(f"2. Kokoro Initialization took: {t_init:.4f} seconds")

# 3. Read text
with open("previews/demo-zh.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 4. Measure Phonemization time
t0 = time.time()
phonemes = kokoro.tokenizer.phonemize(text, lang="en-us")
t_phonemes = time.time() - t0
print(f"3. Phonemization (text to pinyin/phonemes) took: {t_phonemes:.4f} seconds")
print(f"   Phonemes length: {len(phonemes)}")
print(f"   Phonemes string: {phonemes[:100]}...")

# 5. Measure Inference and Post-processing time (GPU by default, unless ONNX_PROVIDER is set)
t0 = time.time()
audio, sr = kokoro.create(text, voice="zf_xiaobei", lang="en-us")
t_inference = time.time() - t0
print(f"4. Audio generation (Inference + Trimming) took: {t_inference:.4f} seconds")
