#!/usr/bin/env python3
import random
import pathlib
import requests

# ---- config ----------------------------------------------------
FAKE_DIR  = pathlib.Path("archive/KAGGLE/AUDIO/FAKE")
API_URL   = "http://localhost:5000/predict"   # adjust if you changed host/port
# ----------------------------------------------------------------

# 1) choose a *.wav from the FAKE directory
wav_files = list(FAKE_DIR.glob("*.wav"))
if not wav_files:
    raise FileNotFoundError(f"No .wav files in {FAKE_DIR!s}")

wav_path = random.choice(wav_files)
print(f"Sending {wav_path.name} …")

# 2) send it via multipart/form-data
with wav_path.open("rb") as f:
    files = {"file": (wav_path.name, f, "audio/wav")}
    resp = requests.post(API_URL, files=files, timeout=30)

# 3) show the response
try:
    resp.raise_for_status()
    print("Server reply:", resp.json())   # e.g. {'label': 'AI'}
except Exception as e:
    print("Request failed:", e, "\nRaw body:", resp.text)