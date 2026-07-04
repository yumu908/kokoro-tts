import subprocess
import time

print("Starting TTS synthesis in the background...")
# Start a synthesis
p = subprocess.Popen([r"venv\python.exe", "-m", "kokoro_tts", "previews/demo.txt", "output.wav", "--voice", "af_sarah"])

# Check nvidia-smi in a loop for 8 seconds to capture python.exe
captured = False
for i in range(16):
    time.sleep(0.5)
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if "python" in res.stdout.lower():
        print(f"\nCaptured python in nvidia-smi at check {i} (time elapsed: {(i+1)*0.5}s):")
        # Print the process list part of nvidia-smi
        lines = res.stdout.split("\n")
        idx = [idx for idx, line in enumerate(lines) if "Processes:" in line]
        if idx:
            print("\n".join(lines[idx[0]:]))
        else:
            print(res.stdout)
        captured = True
        break

if not captured:
    print("\nCould not capture python in nvidia-smi. It might have started and finished too quickly.")

p.wait()
print("\nTTS synthesis completed.")
