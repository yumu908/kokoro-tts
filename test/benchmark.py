import time
import os
import onnxruntime as ort
from kokoro_onnx import Kokoro

model_path = "kokoro-v1.0.onnx"
voices_path = "voices-v1.0.bin"

# 1. Run with CPU
print("--- Initializing CPU Session ---")
os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
kokoro_cpu = Kokoro(model_path, voices_path)
t0 = time.time()
audio, sr = kokoro_cpu.create("Hello world! This is a test of the CPU performance. We will synthesize a slightly longer sentence to make the difference clear and measurable.", voice="af_sarah", speed=1.0)
cpu_time = time.time() - t0
print(f"CPU synthesis time: {cpu_time:.4f} seconds")

# 2. Run with GPU (CUDA)
print("\n--- Initializing GPU (CUDA) Session ---")
os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
kokoro_gpu = Kokoro(model_path, voices_path)
# Warm up GPU (first run on GPU always has compilation overhead)
kokoro_gpu.create("Warm up", voice="af_sarah", speed=1.0)
t0 = time.time()
audio, sr = kokoro_gpu.create("Hello world! This is a test of the GPU performance. We will synthesize a slightly longer sentence to make the difference clear and measurable.", voice="af_sarah", speed=1.0)
gpu_time = time.time() - t0
print(f"GPU synthesis time: {gpu_time:.4f} seconds")

print(f"\nSpeedup: {cpu_time / gpu_time:.2f}x faster on GPU!")
