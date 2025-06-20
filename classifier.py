import numpy as np
import scipy.io.wavfile as wav

beta = np.load("beta11.npy")

func = lambda x: 1.0 / (1.0 + np.exp(-np.dot(x, beta)))

def getRecording(wavFile):
    fs, recording = wav.read(wavFile)
    # print(recording.shape)
    recording = recording.flatten()

    # ─── Step 2: Compute and plot FFT ────────────────────────────────
    # Perform Fast Fourier Transform
    fft_spectrum = np.fft.rfft(recording)
    frequencies = np.fft.rfftfreq(len(recording), d=1/fs)
    magnitude = np.abs(fft_spectrum)

    max = np.argmax(magnitude)
    magnitude = magnitude / magnitude[max]

    return frequencies, magnitude

def classify(file1, file2):
    freq1, mag1 = getRecording(file1)
    freq2, mag2 = getRecording(file2)
    xi = np.abs(mag2 - mag1)
    max = np.argmax(xi)
    xi = xi / xi[max]
    f = func(xi)
    if (f < 0.75):
        return "Different"
    return "Same"