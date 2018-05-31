import numpy as Numpy
import lib as Lib
import sounddevice as SoundDevice
import matplotlib.pyplot as Plot


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
        frequencies = Numpy.full(Lib.CHUNK_SIZE, 1500) if(nonoise == 1) \
            else Numpy.full(Lib.CHUNK_SIZE, 2500)
        t = Numpy.arange(Lib.TIME_PER_CHUNK * Lib.SAMPLES_PER_SEC)

        signal = Numpy.zeros(0)

        for f in frequencies:
            for e in vector:
                if e == 0:
                    signal = Numpy.sin(2 * Numpy.pi * t *
                                       f / Lib.SAMPLES_PER_SEC)
                else:
                    signal = 5 * Numpy.sin(2 * Numpy.pi *
                                           t * f / Lib.SAMPLES_PER_SEC)

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
        for i in range(int(Numpy.floor(record.size - (Lib.NUMBER_DATA_SAMPLES + Lib.NUMBER_NOISE_SAMPLES)))):
            dotProduct = Numpy.dot(noiseToSyncOn,
                                   record[i:Lib.NUMBER_NOISE_SAMPLES + i])
            if (dotProduct > maxDotProduct):
                maxDotProduct = dotProduct
                index = i

        begin = index + int(Lib.NUMBER_NOISE_SAMPLES)
        # begin=181815
        end = begin + int(Lib.NUMBER_DATA_SAMPLES)

        bla = record[begin:end]
        Plot.plot(1.3 * record)
        Plot.plot(Numpy.concatenate(
            [Numpy.zeros(begin), bla, Numpy.zeros(int(end - (Lib.NUMBER_DATA_SAMPLES + Lib.NUMBER_NOISE_SAMPLES)))]))
        Plot.show()

        return bla

    def decodeSignalToBitVectors(self, signal, nonoise):
        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])
        dots = []

        # i = 0
        for chunk in chunks:
            # Plot.plot(1.3 * signal)
            # Plot.plot(Numpy.concatenate(
            #     [Numpy.zeros(i * Lib.ELEMENTS_PER_CHUNK), chunk, Numpy.zeros(len(signal) - (i + 1) * Lib.ELEMENTS_PER_CHUNK)]))
            # Plot.show()
            # i += 1
            dots.append(
                self.decodeSignalChunkToBitVector(chunk, nonoise))

        myMid = Numpy.sum(dots) / len(dots)

        myBits = list(map(lambda x: [0] if x < myMid else[1], dots))

        return myBits

    def decodeSignalChunkToBitVector(self, chunk, nonoise):
        f = 1500 if (nonoise == 1) else 2500
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)

        mysin = Numpy.sin(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)

        dot = Numpy.dot(chunk, mysin)

        return dot
