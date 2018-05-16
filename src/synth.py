import numpy as Numpy
import sounddevice as SoundDevice
import matplotlib.pyplot as Plot
import lib as Lib
import noisedeux as Noise


class Synthesizer:
    def __init__(self):
        pass

    def detectNoise(self):
        SoundDevice.default.channels = 1
        record = SoundDevice.rec(Lib.SAMPLES_PER_SEC * Lib.NOISE_DETECTION_TIME,
                                 Lib.SAMPLES_PER_SEC,
                                 blocking=True)[:, 0]
        SoundDevice.wait()

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

        return 100*Numpy.random.normal(0, 1, Lib.NUMBER_NOISE_SAMPLES)

    def generateCompleteSignal(self, array, nonoise):
        signal = Numpy.zeros(0)

        savedSignalDict = {}

        i = 0
        for a in array:
            if (repr(a) in savedSignalDict):
                signal = Numpy.concatenate(
                    [signal, savedSignalDict.get(repr(a))])
            else:
                inter = self.generateVectorSignal(a, nonoise)
                savedSignalDict[repr(a)] = inter
                signal = Numpy.concatenate([signal, inter])

            if (i == len(array)-300):
                print("Prepare your ears !")
            i = i + 1

        return signal

    def computeFrequencies(self, vector, lowerFrequencyBound):
        frequencies = []

        for i in range(0, Lib.CHUNK_SIZE):
            if (vector[i] == 1):
                frequencies.append(lowerFrequencyBound +
                                   Lib.FREQUENCY_STEP * (i + 1))
            else:
                frequencies.append(-(lowerFrequencyBound +
                                     Lib.FREQUENCY_STEP * (i + 1)))

        return frequencies

    def generateVectorSignal(self, vector, nonoise):
        # Compute the frequencies
        freqs = self.computeFrequencies(vector, Lib.LOWER_LOW_FREQUENCY_BOUND) if (nonoise == 1) \
            else self.computeFrequencies(vector, Lib.LOWER_UPPER_FREQUENCY_BOUND)

        # Prepare the sinuses
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        signal = Numpy.zeros(t.shape)

        for f in freqs:
            signal = signal + 10 * \
                Numpy.sin(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)

        return signal

    def recordSignal(self):
        SoundDevice.default.channels = 1

        return SoundDevice.rec(
            int(Numpy.ceil(Lib.RECORDING_SAMPLES_TOTAL)), Lib.SAMPLES_PER_SEC, blocking=True)[:, 0]

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

        begin = index + Lib.NUMBER_NOISE_SAMPLES
        end = begin + Lib.NUMBER_DATA_SAMPLES

        return record[begin:end]

    def projectSignalChunkOnBasis(self, signalChunk, sinus):
        resultArray = []

        i = 0
        for s in sinus:
            dotProduct = Numpy.dot(s, signalChunk)
            print(dotProduct)
            resultArray.append(1 if (dotProduct >= 0) else 0)

            i = i + 1

        return resultArray

    def decodeSignalToBitVectors(self, signal, nonoise):
        # Compute the basis
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        sinus = Numpy.zeros([Lib.CHUNK_SIZE, len(t)])

        for i in range(0, Lib.CHUNK_SIZE):
            if (nonoise == 1):
                f = Lib.LOWER_LOW_FREQUENCY_BOUND + \
                    Lib.FREQUENCY_STEP * (i + 1)
            else:
                f = Lib.LOWER_UPPER_FREQUENCY_BOUND + \
                    Lib.FREQUENCY_STEP * (i + 1)

            sinus[i, :] = 100*Numpy.sin(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)

        # Compute the chunks corresponding to the vectors and project them on the basis.
        # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        chunks = [signal[i:i + Lib.ELEMENTS_PER_CHUNK]
                  for i in range(0, len(signal), Lib.ELEMENTS_PER_CHUNK)]

        i = 0
        for chunk in chunks:
            chunks[i] = self.projectSignalChunkOnBasis(chunk, sinus)

            i += 1

        return chunks
