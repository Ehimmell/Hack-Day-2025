import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
from scipy.optimize import minimize
import scipy.signal as signal
import matplotlib.pyplot as plt

def getRecording():
    # ─── Step 1: Record audio ────────────────────────────────────────
    duration = 0.5  # seconds
    fs = 44100    # sample rate (Hz)

    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Done recording.")

    # Save the recording
    recording = recording.flatten()  # make 1D

    cutoff = 1000
    nyquist = fs
    cutoff /= nyquist
    b, a = butter(1, cutoff, btype='low', analog=False, fs=fs, output='ba')
    
    recording = lfilter(b, a, recording)

    # ─── Step 2: Compute and plot FFT ────────────────────────────────
    # Perform Fast Fourier Transform
    fft_spectrum = np.fft.rfft(recording)
    frequencies = np.fft.rfftfreq(len(recording), d=1/fs)
    magnitude = np.abs(fft_spectrum)
    return frequencies, magnitude

# Plot frequency spectrum
plt.figure(figsize=(12, 6))

frequencies, magnitude = getRecording()
frequencies2, magnitude2 = getRecording()

max = np.argmax(magnitude)
magnitude = magnitude / magnitude[max]

max2 = np.argmax(magnitude2)
magnitude2 = magnitude2 / magnitude2[max2]

finiteDifferenceDerivs = (magnitude2 - magnitude) / 0.5

maxDeriv = np.argmax(finiteDifferenceDerivs)
finiteDifferenceDerivs = finiteDifferenceDerivs / finiteDifferenceDerivs[maxDeriv]

cleanDerivs = np.where(finiteDifferenceDerivs ** 2 > 0.5, 1, 0.0)

plt.plot(frequencies, magnitude, label="Unfiltered Magnitude Plot 1")
plt.plot(frequencies2, magnitude2, label="Unfiltered Magnitude Plot 2")
# plt.plot(frequencies, finiteDifferenceDerivs, label="Derivs")
plt.plot(frequencies, cleanDerivs, label="Derivs Cleansed")


plt.legend()

plt.title("Frequency Spectrum of Recorded Audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.xlim(0, 5000)  # Show only up to 5 kHz
plt.tight_layout()
plt.show()