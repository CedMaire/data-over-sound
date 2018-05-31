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
        sync=self.createWhiteNoise()
        savedSignalDict = {}

        i=0
        for a in array:
            if (i%500==0):
                signal=Numpy.concatenate([signal,Numpy.zeros(Lib.SAMPLES_PER_SEC),sync])
            if (repr(a) in savedSignalDict):
                signal = Numpy.concatenate(
                    [signal, savedSignalDict.get(repr(a))])
            else:
                inter = self.generateVectorSignal(a, nonoise)
                savedSignalDict[repr(a)] = inter
                signal = Numpy.concatenate([signal, inter])
            i=i+1

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

    def extractDataSignal(self, record,last):
        noiseToSyncOn = self.createWhiteNoise()

        maxDotProduct = 0
        index = 0
        for i in range(0, 4*44100):
            dotProduct = Numpy.dot(noiseToSyncOn,
                                   record[i:Lib.NUMBER_NOISE_SAMPLES + i])
            if (dotProduct > maxDotProduct):
                maxDotProduct = dotProduct
                index = i

        begin = index + int(Lib.NUMBER_NOISE_SAMPLES)
        print(begin)
        end = begin + int(500*Lib.ELEMENTS_PER_CHUNK)
        if(last):
            end=begin+int(40*Lib.ELEMENTS_PER_CHUNK)
        bla = record[begin:end]
        print("bla",len(bla))
        Plot.plot(1.5 * record)
        Plot.plot(Numpy.concatenate([Numpy.zeros(begin), bla, Numpy.zeros(end - begin)]))
        Plot.show()

        return bla

    def decodeSignalToBitVectors(self, signal, nonoise):
        #chunks = [signal[i:i + Lib.SAMPLES_PER_CHUNK]
        #          for i in range(0, len(signal), Lib.SAMPLES_PER_CHUNK)]

        print("signal",len(signal))
        signal1=self.extractDataSignal(signal,False)
        print("signal1",len(signal1))
        signal2=self.extractDataSignal(signal[len(signal1):len(signal)],False)
        signal3=self.extractDataSignal(signal[len(signal2):len(signal)],False)
        signal4=self.extractDataSignal(signal[len(signal3):len(signal)],True)

        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])
        bitVectors = Numpy.zeros((2040,1))
        i=0
        debug=False
        for chunk in chunks:
            print("CHUNK NB",i)
            if (i>728 or i<5):
                debug=True
            bitVectors[i]=self.decodeSignalChunkToBitVector(chunk, nonoise,debug)
            print(bitVectors[i])
            debug=False
            i=1+i
        return bitVectors

    def decodeSignalChunkToBitVector(self, chunk, nonoise,debug):

        w = Numpy.abs(Numpy.fft.fft(chunk[1800:2000]))
        f = Numpy.abs(Numpy.fft.fftfreq(len(w), 1 / Lib.SAMPLES_PER_SEC))

        #if(debug):
            #Plot.plot(chunk)
            #Plot.show()
            #Plot.plot(f, w)
            #Plot.show()
        #print(w.shape, f.shape)
        #print(w)
        #print(f)
        p1=w[9]+w[10]
        p2=w[12]+w[13]
        print(p1,p2)
        if (p1>p2):
            return [int(1)]
        else :
            return [int(0)]

        '''
        peaks = np.empty(2*ones)
        i = 0
        for x in range(2*ones):
            idx = np.argmax(np.abs(w))
            freq = f[idx]
            freq_in_hertz = abs(freq * frequence)
            peaks[i]=freq_in_hertz
            w = np.delete(w, idx)
            idx = np.argmax(np.abs(w))
            w = np.delete(w, idx)
            i+=1
        peaks=np.sort(peaks)
        return peaks
        '''


if __name__ == "__main__":
    synthesizer = Synthesizer()

    synthesizer.decodeSignalChunkToBitVector(
        Numpy.sin(2 * Numpy.pi * Numpy.arange(Lib.SAMPLES_PER_CHUNK) * 1500 / Lib.SAMPLES_PER_SEC), 1)
