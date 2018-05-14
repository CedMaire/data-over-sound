import numpy as Numpy
import lib as Lib
import sounddevice as SoundDevice
from scipy.io import wavfile as WavFile
import os as OS


class NoiseGenerator:
    def __init__(self):
        pass

    def fftNoise(self, f):
        f = Numpy.array(f, dtype="complex")
        Np = int((len(f) - 1) / 2)

        phases = Numpy.random.rand(Np) * 2 * Numpy.pi
        phases = Numpy.cos(phases) + 1j * Numpy.sin(phases)

        f[1:Np+1] *= phases
        f[-1:-1 - Np:-1] = Numpy.conj(f[1:Np + 1])

        return Numpy.fft.ifft(f).real

    def generateBandLimitedNoise(self, min_freq, max_freq, samples=1024, samplerate=1):
        freqs = Numpy.abs(Numpy.fft.fftfreq(samples, 1 / samplerate))
        f = Numpy.zeros(samples)
        idx = Numpy.where(Numpy.logical_or(
            freqs < min_freq, freqs > max_freq))[0]
        f[idx] = 1

        return self.fftNoise(f)


if __name__ == '__main__':
    dirname, _ = OS.path.split(OS.path.abspath(__file__))
    dirname += "/../sound/"

    noise1 = NoiseGenerator().generateBandLimitedNoise(
        Lib.FREE_FREQ_MIN,
        Lib.FREE_FREQ_MAX,
        int(Lib.SAMPLES_PER_SEC * (Lib.TIME_PER_CHUNK + Lib.NOISE_TIME) * 30),
        Lib.SAMPLES_PER_SEC) * Lib.NOISE_AMPLIFIER

    WavFile.write(dirname + "noise_" + repr(Lib.FREE_FREQ_MIN) + "-" +
                  repr(Lib.FREE_FREQ_MAX) + ".wav", Lib.SAMPLES_PER_SEC, noise1)

    # SoundDevice.play(noise1)
    # SoundDevice.wait()
