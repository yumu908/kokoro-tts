import os
import onnxruntime as ort
from kokoro_onnx import Kokoro

model_path = "kokoro-v1.0.onnx"
voices_path = "voices-v1.0.bin"

# 1. Print available providers
print("Available providers on system:", ort.get_available_providers())

# 2. Initialize Kokoro with default settings (should use GPU if available)
print("\n--- Initializing Kokoro ---")
kokoro = Kokoro(model_path, voices_path)

# 3. Check active providers in the session
print("Active providers in Kokoro session:", kokoro.sess.get_providers())
print("Preferred providers passed to session:", kokoro.sess.get_providers())
