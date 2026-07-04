import time
from kokoro_onnx import Kokoro
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
print("Starting 50 synthesis loops to keep GPU busy...")
for i in range(50):
    # This will run model inference 50 times
    kokoro.create("Hello world, this is loop number " + str(i) + ". We are keeping the GPU busy to capture it in nvidia-smi.", voice="af_sarah")
print("Finished loops.")
