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

        return 100 * Numpy.random.normal(0, 1, Lib.NUMBER_NOISE_SAMPLES)

    def generateCompleteSignal(self, array, nonoise):
        sync = self.createWhiteNoise()
        zeros = Numpy.zeros(Lib.SAMPLES_PER_SEC)
        sin0 = self.generateVectorSignal(Numpy.array([0]), nonoise)
        sin1 = self.generateVectorSignal(Numpy.array([1]), nonoise)
        sins = Numpy.zeros([2, len(sin0)])
        sins[0, :] = sin0
        sins[1, :] = sin1  # c'est moche mais c'est pour matcher avec votre truc
        return Numpy.ravel(sins[array, :])
        # array =
        # array1 = array[0:510]
        # array2 = array[510:1020]
        # array3 = array[1020:1530]
        # array4 = array[1530:2040]
        # sins1 = sins[array1, :].reshape([-1])
        # sins2 = sins[array2, :].reshape([-1])
        # sins3 = sins[array3, :].reshape([-1])
        # sins4 = sins[array4, :].reshape([-1])
        # return Numpy.concatenate([sync, sins1, zeros, sync, sins2, zeros, sync, sins3, zeros, sync, sins4])

        # signal = Numpy.zeros(0)
        #
        # savedSignalDict = {}
        #
        # i = 0
        # for a in array:
        #     if (repr(a) in savedSignalDict):
        #         signal = Numpy.concatenate(
        #             [signal, savedSignalDict.get(repr(a))])
        #     else:
        #         inter = self.generateVectorSignal(a, nonoise)
        #         savedSignalDict[repr(a)] = inter
        #         signal = Numpy.concatenate([signal, inter])
        #
        #     if (i == len(array) - 300):
        #         print("Prepare your ears !")
        #     i = i + 1
        #
        # return signal


    def generateVectorSignal(self, vector, nonoise):
        frequencies = []

        for i in range(0, Lib.CHUNK_SIZE):
            if (vector[i] == 1):
                if (nonoise == 1):
                    frequencies.append(int(1021))
                    # Lib.LOWER_LOW_FREQUENCY_BOUND + Lib.FREQUENCY_STEP * (i + 1)))
                else:
                    frequencies.append(int(2027))
                    # Lib.UPPER_LOW_FREQUENCY_BOUND + Lib.FREQUENCY_STEP * (i + 1)))
            else:
                if (nonoise == 1):
                    frequencies.append(int(1913))
                    # Lib.LOWER_LOW_FREQUENCY_BOUND + Lib.FREQUENCY_STEP * (i + 1)))
                else:
                    frequencies.append(int(2789))
                    # Lib.UPPER_LOW_FREQUENCY_BOUND + Lib.FREQUENCY_STEP * (i + 1)))
        t = Numpy.arange(Lib.TIME_PER_CHUNK * Lib.SAMPLES_PER_SEC)
        # print(t)
        # print(frequencies)
        signal = Numpy.zeros(0)

        for f in frequencies:
            signal = Numpy.sin(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)

        return signal

        # # Compute the frequencies
        # freqs = self.computeFrequencies(vector, Lib.LOWER_LOW_FREQUENCY_BOUND) if (nonoise == 1) \
        #     else self.computeFrequencies(vector, Lib.LOWER_UPPER_FREQUENCY_BOUND)
        #
        # # Prepare the sinuses
        # t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        # signal = Numpy.zeros(t.shape)
        #
        # for f in freqs:
        #     signal = signal + 15 * \
        #         Numpy.sin(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)
        #
        # return signal

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
        endsearch = 4*44100
        for i in range(0, endsearch):
            dotProduct = Numpy.dot(noiseToSyncOn,
                                   record[i:Lib.NUMBER_NOISE_SAMPLES + i])
            if (dotProduct > maxDotProduct):
                maxDotProduct = dotProduct
                index = i

        begin = index + int(Lib.NUMBER_NOISE_SAMPLES)
        print(begin)
        end = begin + int(51*Lib.ELEMENTS_PER_CHUNK)
        bla = record[begin:end]
        plot = bla
        for i in range(1, 40):
            begin = end + Lib.SAMPLES_PER_SEC
            end = begin + int(51*Lib.ELEMENTS_PER_CHUNK)
            bla = Numpy.concatenate([bla, record[begin:end]])
            plot = Numpy.concatenate([plot, Numpy.zeros(Lib.SAMPLES_PER_SEC), bla])



        Plot.plot(1.5 * record)
        Plot.plot(Numpy.concatenate([Numpy.zeros(begin), plot]))
        Plot.show()

        return bla
        # noiseToSyncOn = self.createWhiteNoise()
        #
        # index = 0
        # tmp = {}
        # print("Start to compute dot products for noise indexes")
        # for i in range(int(Numpy.floor(record.size - (Lib.NUMBER_DATA_SAMPLES + Lib.NUMBER_NOISE_SAMPLES)))):
        #     dotProduct = Numpy.dot(
        #         noiseToSyncOn, record[i:Lib.NUMBER_NOISE_SAMPLES + i])
        #     tmp[i] = dotProduct
        #
        # biggestIndex = []
        # for x in range (40):
        #     maxDotProduct = 0
        #     for key, val in tmp.items():
        #         if (val > maxDotProduct):
        #             maxDotProduct = val
        #             a = key
        #     biggestIndex.append(a)
        #     del tmp[a]
        #
        # print("indexes : ",biggestIndex)
        #
        # dataSignal = []
        # biggestIndex2 = Numpy.sort(biggestIndex)
        # for x in range(len(biggestIndex2)):
        #     begin = biggestIndex2[x] + Lib.NUMBER_NOISE_SAMPLES
        #     if x < len(biggestIndex2)-1:
        #         end = biggestIndex2[x+1]
        #     else : end = begin + int(Numpy.ceil(Lib.NUMBER_DATA_SAMPLES))
        #     #print(x,"ième concatenation")
        #     if x==39:
        #         dataSignal = Numpy.concatenate([dataSignal, record[begin:end], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
        #     else : dataSignal = Numpy.concatenate([dataSignal, record[begin:end]])
        # print("length of datasignal : ",len(dataSignal)) #786817 962246
        # #print(dataSignal)
        #
        # return dataSignal
        # begin = index + Lib.NUMBER_NOISE_SAMPLES
        # end = begin + int(Numpy.ceil(Lib.NUMBER_DATA_SAMPLES / 40))
        #
        # return record[begin:end], end

    def decodeSignalToBitVectors(self, signal, nonoise):
        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])

        bitVectors = []
        for chunk in chunks:
            bitVectors.append(
                self.decodeSignalChunkToBitVector(chunk, nonoise))
        return bitVectors

    def decodeSignalChunkToBitVector(self, chunk, nonoise):
        w = Numpy.abs(Numpy.fft.fft(chunk[1800:2000]))
        f = Numpy.abs(Numpy.fft.fftfreq(len(w), 1 / Lib.SAMPLES_PER_SEC))

        idx = Numpy.argmax(w)
        freq = f[idx]
        print(freq)
        # Plot.plot(f, w)
        # Plot.show()
        if nonoise ==1:
            if (Numpy.abs(1913-freq)<Numpy.abs(1021-freq)):
                return [0]
            else :
                return [1]
        else :
            if (Numpy.abs(2789-freq)<Numpy.abs(2027-freq)):
                return [0]
            else :
                return [1]
