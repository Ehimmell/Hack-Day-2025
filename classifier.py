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
    frequencies = np.fft.rfftfreq(len(recording), d=1 / fs)
    magnitude = np.abs(fft_spectrum)

    max = np.argmax(magnitude)
    magnitude = magnitude / magnitude[max]

    return frequencies, magnitude


def classify(file1, file2):
    freq1, mag1 = getRecording(file1)
    freq2, mag2 = getRecording(file2)

    segments = np.floor(min(len(mag1), len(mag2)) / len(beta))
    fSum = 0
    for i in range(int(segments)):
        m1 = mag1[i * len(beta): (i + 1) * len(beta)]
        m2 = mag2[i * len(beta): (i + 1) * len(beta)]
        xi = np.abs(m2 - m1)
        maxArg = np.argmax(xi)
        xi = xi / xi[maxArg]
        f = func(xi)
        fSum += f
    fSum /= segments

    if (fSum < 0.75):
        return "Different"
    return "Same"