import numpy as Numpy
import lib as Lib
import sounddevice as SoundDevice
import matplotlib.pyplot as Plot
from scipy.signal import find_peaks_cwt
from scipy.interpolate import interp1d
import peakutils


class Synthesizer:
    def __init__(self):
        pass

    def detectNoise(self):
        return 2
        record = SoundDevice.rec(Lib.SAMPLES_PER_SEC * Lib.NOISE_DETECTION_TIME,
                                 Lib.SAMPLES_PER_SEC,
                                 blocking=True,
                                 channels=1)[:, 0]

        recordfft = Numpy.fft.fft(record)

        sum1000 = Numpy.sum(Numpy.abs(recordfft[1000:2000]))
        sum2000 = Numpy.sum(Numpy.abs(recordfft[2000:3000]))

        if (sum1000 > sum2000):
            print("Noise-free Frequencies: [2000:3000]")
            return 2
        else:
            print("Noise-free Frequencies: [1000:2000]")
            return 1

    def createWhiteNoise(self):
        Numpy.random.seed(Lib.NOISE_SEED)

        return Numpy.random.normal(0, 1, Lib.NUMBER_NOISE_SAMPLES)

    def generateCompleteSignal(self, array, nonoise):
        signal = Numpy.zeros(0)
        savedSignalDict = {}

        for a in array:
            if (repr(a) in savedSignalDict):
                signal = Numpy.concatenate(
                    [signal, savedSignalDict.get(repr(a))])
            else:
                inter = self.generateVectorSignal(a, nonoise)
                savedSignalDict[repr(a)] = inter
                signal = Numpy.concatenate([signal, inter])

        return signal

    def generateVectorSignal(self, vector, nonoise):
        frequencies = []

        for i in range(0, Lib.CHUNK_SIZE):
            if (vector[i] == 1):
                if (nonoise == 1):
                    frequencies.append(int(1027))
                        #Lib.LOWER_LOW_FREQUENCY_BOUND + Lib.FREQUENCY_STEP * (i + 1)))
                else:
                    frequencies.append(int(2027))
                        #Lib.UPPER_LOW_FREQUENCY_BOUND + Lib.FREQUENCY_STEP * (i + 1)))
            else :
                if (nonoise == 1):
                    frequencies.append(int(1234))
                        #Lib.LOWER_LOW_FREQUENCY_BOUND + Lib.FREQUENCY_STEP * (i + 1)))
                else:
                    frequencies.append(int(2789))
                        #Lib.UPPER_LOW_FREQUENCY_BOUND + Lib.FREQUENCY_STEP * (i + 1)))
        t = Numpy.arange(Lib.TIME_PER_CHUNK * Lib.SAMPLES_PER_SEC)
        #print(t)
        #print(frequencies)
        signal = Numpy.zeros(0)

        for f in frequencies:
            signal = Numpy.sin(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)

        return signal

    def recordSignal(self):
        recording = SoundDevice.rec(int(Lib.RECORDING_SAMPLES_TOTAL),
                                    Lib.SAMPLES_PER_SEC,
                                    blocking=True,
                                    channels=1)[:, 0]

        return recording

    def extractDataSignal(self, record):
        noiseToSyncOn = self.createWhiteNoise()

        maxDotProduct = 0
        index = 0
        for i in range(4*44100):
            dotProduct = Numpy.dot(noiseToSyncOn,
                                   record[i:Lib.NUMBER_NOISE_SAMPLES + i])
            if (dotProduct > maxDotProduct):
                maxDotProduct = dotProduct
                index = i

        begin = index + int(Lib.NUMBER_NOISE_SAMPLES)
        end = begin + int(Lib.NUMBER_DATA_SAMPLES*(500/2040))
        bla = record[begin:end]
        for i in range(4):
            begin = end + int(Lib.NUMBER_NOISE_SAMPLES)
            end = begin + int(Lib.NUMBER_DATA_SAMPLES* (500/2040))
            bla = bla.append(record[begin:end])
        Plot.plot(1.5 * record)
        Plot.plot(Numpy.concatenate([Numpy.zeros(begin), bla, Numpy.zeros(end - begin)]))
        Plot.show()

        return bla

    def decodeSignalToBitVectors(self, signal, nonoise):
        #chunks = [signal[i:i + Lib.SAMPLES_PER_CHUNK]
        #          for i in range(0, len(signal), Lib.SAMPLES_PER_CHUNK)]
        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])
        bitVectors = []
        for chunk in chunks:
            bitVectors.append(
                self.decodeSignalChunkToBitVector(chunk, nonoise))

        return bitVectors

    def decodeSignalChunkToBitVector(self, chunk, nonoise):
        # Plot.plot(chunk)
        # Plot.show()
        w = Numpy.abs(Numpy.fft.fft(chunk[1800:2000]))
        f = Numpy.abs(Numpy.fft.fftfreq(len(w), 1 / Lib.SAMPLES_PER_SEC))

        idx = Numpy.argmax(w)
        freq = f[idx]
        print(freq)

        # Plot.plot(f, w)
        # Plot.show()

        if (Numpy.abs(2789-freq)<Numpy.abs(2027-freq)):
            return [0]
        else :
            return [1]
