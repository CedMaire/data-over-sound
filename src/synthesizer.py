import numpy as Numpy
import lib as Lib
import matplotlib.pyplot as Plot
import sounddevice as SoundDevice
from scipy import signal as Signal
import heapq as HeapQ


class Synthesizer:
    def __init__(self):
        pass

    def createWhiteNoise(self):
        Numpy.random.seed(Lib.NOISE_SEED)

        return Numpy.random.normal(0, 1, Lib.NUMBER_NOISE_SAMPLES)

    def generateCompleteSignal(self, vectorArray, nonoise):
        completeSignal = self.createWhiteNoise()
        savedSignalDict = {}

        for index, vector in enumerate(vectorArray):
            if ((index % Lib.VECTORS_CHUNKS_SIZE_BEFORE_ZEROS) == 0):
                completeSignal = Numpy.concatenate(
                    [completeSignal, Numpy.zeros(Lib.NUMBER_ZEROS_SAMPLES)])
            if (repr(vector) in savedSignalDict):
                completeSignal = Numpy.concatenate(
                    [completeSignal, savedSignalDict.get(repr(vector))])
            else:
                vectorSignal = self.generateVectorSignal(vector, nonoise)
                completeSignal = Numpy.concatenate(
                    [completeSignal, vectorSignal])
                savedSignalDict[repr(vector)] = vectorSignal

        return completeSignal

    def generateVectorSignal(self, vector, nonoise):
        t = Numpy.arange(Lib.SAMPLES_PER_VECTOR)
        f0 = Lib.FREQ_NONOISE1_0 if (nonoise == 1) else Lib.FREQ_NONOISE2_0
        f1 = Lib.FREQ_NONOISE1_1 if (nonoise == 1) else Lib.FREQ_NONOISE2_1

        mySin0 = Numpy.sin(2 * Numpy.pi * t * f0 / Lib.FS)
        mySin1 = Numpy.sin(2 * Numpy.pi * t * f1 / Lib.FS)

        mySin = mySin0 if (vector[0] == 0) else mySin1

        return mySin

    def extractDataSignal(self, recording):
        noiseToSyncOn = self.createWhiteNoise()

        maxDotProduct = 0
        index = 0

        for i in range(0, len(recording) - Lib.NUMBER_NOISE_SAMPLES - Lib.NUMBER_DATA_SAMPLES - Lib.NUMBER_ZEROS_SAMPLES_TOTAL):
            dotProduct = Numpy.dot(noiseToSyncOn,
                                   recording[i:i + Lib.NUMBER_NOISE_SAMPLES])
            if (dotProduct > maxDotProduct):
                maxDotProduct = dotProduct
                index = i

        begin = index + Lib.NUMBER_NOISE_SAMPLES + Lib.NUMBER_ZEROS_SAMPLES
        end = begin + Lib.NUMBER_DATA_SAMPLES + Lib.NUMBER_ZEROS_SAMPLES_TOTAL

        dataWithZeros = recording[begin:end]
        # Plot.plot(1.2 * recording)
        # Plot.plot(Numpy.concatenate([Numpy.zeros(begin), dataWithZeros, Numpy.zeros(len(
        #     recording) - Lib.NUMBER_NOISE_SAMPLES - Lib.NUMBER_DATA_SAMPLES - (begin - Lib.NUMBER_NOISE_SAMPLES) - Lib.NUMBER_ZEROS_SAMPLES_TOTAL)]))
        # Plot.show()

        data = Numpy.zeros(0)
        for j in range(0, Lib.NUMBER_VECTOR_GROUPS):
            start = j * (Lib.NUMBER_SAMPLES_PER_VECTORS_CHUNK +
                         Lib.NUMBER_ZEROS_SAMPLES - Lib.MAGIC_NUMBER)
            stop = start + Lib.NUMBER_SAMPLES_PER_VECTORS_CHUNK
            data = Numpy.concatenate([data, dataWithZeros[start:stop]])

            # Plot.plot(1.2 * recording)
            # Plot.plot(Numpy.concatenate([Numpy.zeros(start + Lib.NUMBER_NOISE_SAMPLES + (begin - Lib.NUMBER_NOISE_SAMPLES)),
            #                              dataWithZeros[start:stop],
            #                              Numpy.zeros(len(
            #                                  recording) - Lib.NUMBER_NOISE_SAMPLES - (begin - Lib.NUMBER_NOISE_SAMPLES) - start - Lib.NUMBER_SAMPLES_PER_VECTORS_CHUNK)]))
            # Plot.show()

        dataInterleaved = Numpy.zeros(0)
        dataReshaped = data.reshape([-1, Lib.NUMBER_SAMPLES_PER_VECTORS_CHUNK])
        for k, vectorsChunk in enumerate(dataReshaped):
            chunk = vectorsChunk if (k == 0) else Numpy.concatenate(
                [Numpy.zeros(Lib.NUMBER_ZEROS_SAMPLES), vectorsChunk])
            dataInterleaved = Numpy.concatenate([dataInterleaved, chunk])

        Plot.plot(1.2 * recording)
        Plot.plot(Numpy.concatenate([Numpy.zeros(begin),
                                     dataInterleaved,
                                     Numpy.zeros(len(
                                         recording) - Lib.NUMBER_NOISE_SAMPLES - Lib.NUMBER_DATA_SAMPLES - (begin - Lib.NUMBER_NOISE_SAMPLES) - Lib.NUMBER_ZEROS_SAMPLES_TOTAL)]))
        Plot.show()

        return data

    def recordSignal(self):
        recording = SoundDevice.rec(Lib.RECORDING_SAMPLES_TOTAL,
                                    Lib.FS,
                                    blocking=True,
                                    channels=1)[:, 0]

        return recording

    def decodeSignalToBitVectors(self, dataSignal, nonoise):
        chunks = dataSignal.reshape([-1, Lib.SAMPLES_PER_VECTOR])

        decodedVectors = []

        for index, chunk in enumerate(chunks):
            # concat = Numpy.concatenate([Numpy.zeros(index * Lib.SAMPLES_PER_VECTOR),
            #                             chunk, Numpy.zeros(Lib.NUMBER_DATA_SAMPLES - ((index + 1) * Lib.SAMPLES_PER_VECTOR))])
            # Plot.plot(concat)

            if(index > 830):
                boolean = True
            else:
                boolean = False

            decodedVectors.append(
                self.decodeSignalChunkToBitVector(chunk, nonoise, boolean))

        # Plot.show()
        return decodedVectors

    def decodeSignalChunkToBitVector(self, chunk, nonoise, boolean):
        w = Numpy.abs(Numpy.fft.fft(chunk))
        f = Numpy.fft.fftfreq(len(w), 1 / Lib.FS)
        # print(w)
        # print(f)

        if(boolean):
            Plot.plot(f, w)
            Plot.show()
            Plot.plot(chunk)
            Plot.show()
        idx = Numpy.argmax(w[60:95])
        freq = f[idx + 60]

        freq0 = 1297
        freq1 = 1697
        if (Numpy.abs(freq1 - freq) < Numpy.abs(freq0 - freq)):
            return [1]
        else:
            return [0]

        '''
        # wTMP = HeapQ.nlargest(2, w)
        # print(wTMP)

        peaks = Signal.find_peaks_cwt(w, [50, 50], noise_perc=1)
        # print(peaks)

        tuples = list(map(lambda x: (x, w[x]), peaks))
        # print(tuples)
        tuples = HeapQ.nlargest(2, tuples, key=lambda x: x[1])
        # print(tuples)

        Plot.plot(w)
        Plot.show()

        zeroa = 115 if (nonoise == 2) else 65
        zerob = 236 if (nonoise == 2) else 286

        onea = 135 if (nonoise == 2) else 85
        oneb = 216 if(nonoise == 2) else 266

        realzeroa = peaks[Numpy.abs(peaks - zeroa).argmin()]
        realzerob = peaks[Numpy.abs(peaks - zerob).argmin()]
        realonea = peaks[Numpy.abs(peaks - onea).argmin()]
        realoneb = peaks[Numpy.abs(peaks - oneb).argmin()]

        testZero = abs(realzeroa - zeroa) + abs(realzerob - zerob)
        testOne = abs(realonea - onea) + abs(realoneb - oneb)
        print(testZero, testOne)

        return [0] if (testZero < testOne) else [1]
        '''
