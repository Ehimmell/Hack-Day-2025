import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import find_peaks
from scipy.optimize import minimize
import scipy.signal as signal
import matplotlib.pyplot as plt

# ─── Step 1: Record audio ────────────────────────────────────────
duration = 3  # seconds
fs = 44100    # sample rate (Hz)

print("Recording...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("Done recording.")

# Save the recording
recording = recording.flatten()  # make 1D

r = 0.7
arr = []
for i in range(20):
    arr.append(((1 - r)) * (r ** i))

recording = np.convolve(recording, arr)

# wav.write("output.wav", fs, recording.astype(np.float32))

# ─── Step 2: Compute and plot FFT ────────────────────────────────
# Perform Fast Fourier Transform
fft_spectrum = np.fft.rfft(recording)
frequencies = np.fft.rfftfreq(len(recording), d=1/fs)
magnitude = np.abs(fft_spectrum)

# Plot frequency spectrum
plt.figure(figsize=(12, 6))

peaks, _ = find_peaks(magnitude)

max = np.argmax(magnitude)
magnitude = magnitude / magnitude[max]
n = 10
fundamental_freq = frequencies[max]
harmonic_indecies = []

def scoreFundamentalFreq(ff):
    sum = 0
    for i in range(len(frequencies)):
        remainder = frequencies[i] % ff
        distance_to_previous = abs(remainder)
        distance_to_next = abs(ff) - abs(remainder)
        min_dist = min(distance_to_previous, distance_to_next)
        # if (min_dist > 10):
        sum += magnitude[i] * (min_dist / ff) ** 8
    return sum

res = minimize(scoreFundamentalFreq, fundamental_freq, method='Nelder-Mead')
fundamental_freq = res.x

for i in range(n):
    differences = np.abs(frequencies - fundamental_freq * (i + 1))
    harmonic_indecies.append(np.argmin(differences))

plt.plot(frequencies, magnitude, label="Unfiltered Magnitude Plot")

for i in range(len(frequencies)):
    remainder = frequencies[i] % fundamental_freq
    distance_to_previous = abs(remainder)
    distance_to_next = abs(fundamental_freq) - abs(remainder)
    min_dist = min(distance_to_previous, distance_to_next)
    magnitude[i] = magnitude[i] * (1.0 - min_dist / fundamental_freq) ** 2

plt.plot(frequencies, magnitude, label="Filtered Magnitude Plot")
plt.plot(frequencies[peaks], magnitude[peaks], label="Peaks", linestyle="none", marker=".")
plt.plot(frequencies[harmonic_indecies], magnitude[harmonic_indecies], label="Harmonics", linestyle="none", marker="o")

plt.legend()

plt.title("Frequency Spectrum of Recorded Audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.xlim(0, 2000)  # Show only up to 5 kHz
plt.tight_layout()
plt.show()