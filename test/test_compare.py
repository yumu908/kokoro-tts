import time
import os
import onnxruntime as ort
from kokoro_onnx import Kokoro

model_path = "kokoro-v1.0.onnx"
voices_path = "voices-v1.0.bin"

# 1. CPU
print("=== Running on CPU ===")
os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
kokoro_cpu = Kokoro(model_path, voices_path)
# Warm up
kokoro_cpu.create("warm up", voice="af_sarah")
t0 = time.time()
for _ in range(10):
    audio, sr = kokoro_cpu.create("Hello world. This is a benchmark to compare CPU and GPU speeds. We will run it multiple times to get a stable measurement.", voice="af_sarah")
print(f"CPU took: {time.time() - t0:.4f} seconds")

# 2. GPU
print("=== Running on GPU ===")
os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
kokoro_gpu = Kokoro(model_path, voices_path)
# Warm up
kokoro_gpu.create("warm up", voice="af_sarah")
t0 = time.time()
for _ in range(10):
    audio, sr = kokoro_gpu.create("Hello world. This is a benchmark to compare CPU and GPU speeds. We will run it multiple times to get a stable measurement.", voice="af_sarah")
print(f"GPU took: {time.time() - t0:.4f} seconds")
