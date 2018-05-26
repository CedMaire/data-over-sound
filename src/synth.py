import numpy as Numpy
import sounddevice as SoundDevice
import matplotlib.pyplot as Plot
import iodeux as IODeux
import lib as Lib
import coder as Coder
# import noisedeux as Noise


class Synthesizer:
    def __init__(self):
        pass

    def detectNoise(self):
        return 2 # deal with it
        SoundDevice.default.channels = 1
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

    '''def generateCompleteSignal(self, array, nonoise):
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

            if (i == len(array) - 300):
                print("Prepare your ears !")
            i = i + 1

        return signal'''

    def generateCompleteSignal(self, array, nonoise):
        sin0 = self.generateVectorSignal(Numpy.array([0]), nonoise)
        sin1 = self.generateVectorSignal(Numpy.array([1]), nonoise)
        Plot.plot(sin0)
        Plot.plot(sin1)
        Plot.show()
        sins = Numpy.zeros([2,len(sin0)])
        sins[0,:]=sin0
        sins[1,:]=sin1# c'est moche mais c'est pour matcher avec votre truc
        return sins[array,:].reshape([-1])


    '''def computeFrequencies(self, vector, lowerFrequencyBound):
        frequencies = []

        for i in range(0, Lib.CHUNK_SIZE):
            if (vector[i] == 1):
                frequencies.append(lowerFrequencyBound +
                                   Lib.FREQUENCY_STEP * (i + 1))
            else:
                frequencies.append(-(lowerFrequencyBound +
                                     Lib.FREQUENCY_STEP * (i + 1)))

        return frequencies'''
    def computeFrequencies(self, vector, lowerFrequencyBound):
        frequencies = Numpy.ones(vector.shape) * lowerFrequencyBound + Lib.FREQUENCY_STEP
        frequencies[vector == 0] = -1*frequencies[vector == 0]
        return frequencies

    def generateVectorSignal(self, vector, nonoise):
        # Compute the frequencies
        """
            freqs = self.computeFrequencies(vector, Lib.LOWER_LOW_FREQUENCY_BOUND) if (nonoise == 1) \
            else self.computeFrequencies(vector, Lib.LOWER_UPPER_FREQUENCY_BOUND)
        """
        f=1500 if nonoise==1 else 2500
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        signal = Numpy.zeros(t.shape)
        if(vector[0]==1):
            signal=signal+Numpy.sin(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)
        if(vector[0]==0):
            signal=signal+Numpy.sin(2 * Numpy.pi * t * (-f) / Lib.SAMPLES_PER_SEC)
        if(vector[1]==1):
            signal=signal+Numpy.cos(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)
        if(vector[1]==0):
            signal=signal+Numpy.cos(2 * Numpy.pi * t * (-f) / Lib.SAMPLES_PER_SEC)

        return signal

    def recordSignal(self):
        SoundDevice.default.channels = 1

        bla = SoundDevice.rec(int(Numpy.ceil(Lib.RECORDING_SAMPLES_TOTAL)),
                              Lib.SAMPLES_PER_SEC,
                              blocking=True,
                              channels=1)[:, 0]
        Plot.plot(bla)
        Plot.show()
        return bla

    def extractDataSignal(self, record):
        noiseToSyncOn = self.createWhiteNoise()

        maxDotProduct = 0
        index = 0
        """
        for i in range(int(Numpy.floor(record.size - (Lib.NUMBER_DATA_SAMPLES + Lib.NUMBER_NOISE_SAMPLES)))):
            dotProduct = Numpy.dot(noiseToSyncOn,
                                   record[i:Lib.NUMBER_NOISE_SAMPLES + i])
            if (dotProduct > maxDotProduct):
                maxDotProduct = dotProduct
                index = i
        """
        #begin = index + Lib.NUMBER_NOISE_SAMPLES
        begin=181815
        end = begin + Lib.NUMBER_DATA_SAMPLES

        bla = record[begin:end]
        Plot.plot(1.5 * record)
        Plot.plot(Numpy.concatenate([Numpy.zeros(begin), bla, Numpy.zeros(end - begin)]))
        Plot.show()

        return bla

    def decodeur2LEspace(self,signal,nonoise):
        phaseSeeker=128
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        sinus = Numpy.zeros([phaseSeeker, len(t)])

        if (nonoise == 1):
            f = Lib.LOWER_LOW_FREQUENCY_BOUND+Lib.FREQUENCY_STEP
        else:
            f = Lib.LOWER_UPPER_FREQUENCY_BOUND+Lib.FREQUENCY_STEP
        j=0
        for i in range (phaseSeeker):
            sinus[j]=Numpy.sin((2*Numpy.pi*t*f/Lib.SAMPLES_PER_SEC)-1.5+i*0.05)
            j=j+1

        #plot signal versus sin
        #Plot.plot(0.1*Numpy.tile(sinus[0,:], [1,2])[0,:])
        #Plot.plot(signal[:2*Lib.ELEMENTS_PER_CHUNK])
        #Plot.plot(Numpy.zeros(2*Lib.ELEMENTS_PER_CHUNK))
        #Plot.show()

        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])
        dotArray=Numpy.zeros([chunks.shape[0],phaseSeeker])
        i=0
        resultArray=[]
        currphase=0
        for chunk in chunks:
            dotArray[i,:]=chunk @ sinus.T
            #Plot.plot(dotArray[i,:])
            #Plot.show()
            #!! And here starts the fun !!
            min=0
            max=0
            jmax=None
            jmin=None
            line=len(dotArray[i])
            for j in range(phaseSeeker):
                if (dotArray[i][j]>max):
                    max=dotArray[i][j]
                    jmax=j
                elif (dotArray[i][j]<min):
                    min=dotArray[i][j]
                    jmin=j
            jdistmin=self.findClosestIndex(currphase,jmin,jmax,phaseSeeker)
            if (dotArray[i][jdistmin]<0):
                resultArray.append([0])
                currphase=jdistmin
            else:
                resultArray.append([1])
                currphase=jdistmin
            print(currphase)
        i=i+1
        return resultArray

    def findClosestIndex(self,j0,j1,j2,phaseSeeker):
        d1=Numpy.abs(j1-j0)
        if(d1>phaseSeeker/2):
            d1=phaseSeeker-d1
        d2=Numpy.abs(j2-j0)
        if(d2>phaseSeeker/2):
            d2=phaseSeeker-d2
        #print("debug", "d1",d1,"j1",j1,"d2",d2,"j2",j2)
        return j1 if d1<d2 else j2
