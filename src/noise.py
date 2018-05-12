import numpy as np
from scipy.io import wavfile

freqs_stable = [2000,3000]

duration = 5

fs = 44100

num_samples = fs*duration

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_or(freqs<min_freq, freqs>max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

x = band_limited_noise(freqs_stable[0], freqs_stable[1], fs*duration, fs)*100
x = np.int16(x * (2**15 - 1))
wavfile.write("noise[" + str(freqs_stable[0]) + "," + str(freqs_stable[1]) + "].wav", fs, x)


