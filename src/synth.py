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
        freqs = self.computeFrequencies(vector, Lib.LOWER_LOW_FREQUENCY_BOUND) if (nonoise == 1) \
            else self.computeFrequencies(vector, Lib.LOWER_UPPER_FREQUENCY_BOUND)

        # Prepare the sinuses
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        signal = Numpy.zeros(t.shape)

        for f in freqs:
            signal = signal + Numpy.sin(2 * Numpy.pi * t * f / Lib.SAMPLES_PER_SEC)

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

    def projectSignalChunkOnBasis(self, signalChunk, sinus):
        resultArray = []
        i = 0
        maxDot = 0
        """
        Plot.plot(sinus[0])
        Plot.plot(sinus[1])
        Plot.plot(sinus[9])
        Plot.show()
        """
        """
        for s in sinus:
            dotProduct = Numpy.dot(s, signalChunk)
            print(i, dotProduct)
            if(Numpy.abs(dotProduct) > Numpy.abs(maxDot)):
                maxDot = dotProduct
            i = i + 1
        """
        dotProduct = Numpy.dot(sinus[0], signalChunk)
        print(Numpy.abs(dotProduct))
        resultArray.append(1 if (dotProduct >= 0) else 0)
        return resultArray

    '''
    def realDecode(self, dotArray,resultArray):
        print(len(resultArray), len(dotArray))
        index=Numpy.ndarray.tolist(Numpy.arange(len(dotArray)))
        zipped=zip(dotArray,index)
        separate=[]
        block=[]
        for a,b in zipped:
            if a>15:
                block.append(b)
            elif a<=15:
                if len(block) != 0:
                     separate.append(block)
                block=[b]
                separate.append(block)
                block=[]
        separate.append(block)
        flip=True
        print(separate)
        counter=0

        io = IODeux.IODeux()
        coder = Coder.Coder()
        stringRead = io.readFile(Lib.FILENAME_READ)
        encodedVectors = coder.encode(stringRead)


        for s in separate:
            if (len(s)>2 and flip):
                flip=False
                #print("Flipping from" ,s[0], "to", s[len(s)-1])
                for i in s:
                    resultArray[i]=[0] if resultArray[i]==[1] else [1]
            elif (len(s)>2 and not flip):
                #print("NOT Flipping from" ,s[0], "to", s[len(s)-1])
                flip=True
            elif (len(s)==1):
                counter=counter+1
                print(s, resultArray[s[0]], encodedVectors[s[0]], dotArray[s[0]])
        return resultArray

    '''
    '''
    def decodeSignalToBitVectors(self, signal, nonoise):
        # Compute the basis
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        sinus = Numpy.zeros([4000, len(t)])

        for j in range(0, 4000):
            if (nonoise == 1):
                f = Lib.LOWER_LOW_FREQUENCY_BOUND + \
                    Lib.FREQUENCY_STEP
            else:
                f = Lib.LOWER_UPPER_FREQUENCY_BOUND + \
                    Lib.FREQUENCY_STEP

            sinus[j, :] = Numpy.sin((((2*Numpy.pi*t*(f-j/4)/44100))))

        # Compute the chunks corresponding to the vectors and project them on the basis.
        # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        chunks = [signal[i:i + Lib.ELEMENTS_PER_CHUNK]
                  for i in range(0, len(signal), Lib.ELEMENTS_PER_CHUNK)]
        resultArray=[]
        dotArray=[]
        for chunk in chunks:
            dotProduct=Numpy.dot(sinus[0], chunk)
            dotArray.append(Numpy.abs(dotProduct))
            resultArray.append([1] if (dotProduct >= 0) else [0])
            i += 1
        #Plot.plot(dotArray)
        Plot.plot(Numpy.fft.fftfreq(len(dotArray), 1/44100),Numpy.fft.fft(dotArray))
        Plot.show()
        #resultArray=self.realDecode(dotArray,resultArray)
        return resultArray

    '''
    def decodeSignalToBitVectors(self, signal, nonoise):
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        sinus = Numpy.zeros([128, len(t)])

        if (nonoise == 1):
            f = Lib.LOWER_LOW_FREQUENCY_BOUND+Lib.FREQUENCY_STEP
        else:
            f = Lib.LOWER_UPPER_FREQUENCY_BOUND+Lib.FREQUENCY_STEP
        j=0
        for i in range (128):
            sinus[j]=Numpy.sin((2*Numpy.pi*t*f/Lib.SAMPLES_PER_SEC)-1+i*0.05)
            j=j+1

        #plot signal versus sin
        Plot.plot(0.1*Numpy.tile(sinus[0,:], [1,2])[0,:])
        Plot.plot(signal[:2*Lib.ELEMENTS_PER_CHUNK])
        Plot.plot(Numpy.zeros(2*Lib.ELEMENTS_PER_CHUNK))
        Plot.show()

        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])
        dotArray=Numpy.zeros([chunks.shape[0],128])
        i=0
        for chunk in chunks:
            dotArray[i,:]=chunk @ sinus.T
            Plot.plot(dotArray[i,:])
            Plot.show()
            i=i+1

        #Plot.plot(Numpy.abs(dotArray))
        #Plot.plot(Numpy.abs(Numpy.fft.fft(dotArray)))
        #Plot.show()

        resultArray=[]
        #for b in dotArray >= 0:
        #    resultArray.append([1] if b else [0])
        return resultArray

    def Decodeur2LEspace(self,signal,nonoise):
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        sinus = Numpy.zeros([128, len(t)])

        if (nonoise == 1):
            f = Lib.LOWER_LOW_FREQUENCY_BOUND+Lib.FREQUENCY_STEP
        else:
            f = Lib.LOWER_UPPER_FREQUENCY_BOUND+Lib.FREQUENCY_STEP
        j=0
        for i in range (128):
            sinus[j]=Numpy.sin((2*Numpy.pi*t*f/Lib.SAMPLES_PER_SEC)-1.5+i*0.05)
            j=j+1

        #plot signal versus sin
        Plot.plot(0.1*Numpy.tile(sinus[0,:], [1,2])[0,:])
        Plot.plot(signal[:2*Lib.ELEMENTS_PER_CHUNK])
        Plot.plot(Numpy.zeros(2*Lib.ELEMENTS_PER_CHUNK))
        Plot.show()

        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])
        dotArray=Numpy.zeros([chunks.shape[0],128])
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
            for j in range(128):
                if (dotArray[i][j]>max):
                    max=dotArray[i][j]
                    jmax=j
                elif (dotArray[i][j]<min):
                    min=dotArray[i][j]
                    jmin=j
            print(max,jmax,min,jmin)
            jdistmin=self.findClosestIndex(currphase,jmin,jmax)
            if (dotArray[i][jdistmin]<0):
                resultArray.append([0])
                currphase=jdistmin
            else:
                resultArray.append([1])
                currphase=jdistmin
            print(currphase)
        i=i+1
        return resultArray

    def findClosestIndex(self,j0,j1,j2):
        d1=Numpy.abs(j1-j0)
        if(d1>64):
            d1=128-d1
        d2=Numpy.abs(j2-j0)
        print("debug", "j0",j0,"j1",j1,"j2",j2,"d1",d1,"d2",d2)
        if(d2>64):
            d2=128-d2
        return j1 if d1<d2 else j2
