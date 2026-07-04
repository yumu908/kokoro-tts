import subprocess
import time

print("Starting busy_loop.py in the background...")
p = subprocess.Popen([r"venv\python.exe", "busy_loop.py"])

# Wait 3 seconds for it to start and initialize CUDA
time.sleep(3.0)

print("\n--- Running nvidia-smi now ---")
subprocess.run(["nvidia-smi"])

# Wait for it to finish
p.wait()
print("Completed.")
