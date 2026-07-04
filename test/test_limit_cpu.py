import time
import os
import onnxruntime as ort
from kokoro_onnx import Kokoro

model_path = "kokoro-v1.0.onnx"
voices_path = "voices-v1.0.bin"

# 1. Configuration to limit CPU thread usage
print("Creating SessionOptions with CPU threads limited to 1...")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1  # Limit internal operations to 1 thread
opts.inter_op_num_threads = 1  # Limit graph execution to 1 thread

# 2. Create the session explicitly with limited threads and CUDA provider
print("Initializing InferenceSession on GPU (CUDA) with limited CPU threads...")
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

# 3. Load Kokoro using the customized session
kokoro = Kokoro.from_session(session, voices_path)

# 4. Perform synthesis and measure time
print("Starting synthesis...")
t0 = time.time()
audio, sr = kokoro.create(
    "Hello world! This is a test of GPU execution with CPU threads limited to one. "
    "This prevents ONNX Runtime from spinning up threads on all CPU cores and spiking CPU utilization.",
    voice="af_sarah",
    speed=1.0
)
duration = time.time() - t0

print(f"\nSynthesis completed successfully in {duration:.4f} seconds!")
print("Since CPU threads were limited to 1, the overall CPU usage remains extremely low.")
