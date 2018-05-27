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

    def generateCompleteSignal(self, array, nonoise):
        sig00 = self.generateVectorSignal(Numpy.array([0,0]), nonoise)
        sig01 = self.generateVectorSignal(Numpy.array([0,1]), nonoise)
        sig10 = self.generateVectorSignal(Numpy.array([1,0]), nonoise)
        sig11 = self.generateVectorSignal(Numpy.array([1,1]), nonoise)
        sins = Numpy.zeros([4,len(sig00)])
        sins[0,:]=sig00
        sins[1,:]=sig01# c'est moche mais c'est pour matcher avec votre truc
        sins[2,:]=sig10
        sins[3,:]=sig11
        array = Numpy.array(array)
        return sins[array[:,0]*2+array[:,1],:].reshape([-1])

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
            signal=signal+Numpy.sin(2 * Numpy.pi * t * 2053 / Lib.SAMPLES_PER_SEC)
        if(vector[0]==0):
            signal=signal+Numpy.sin(2 * Numpy.pi * t * (-2053) / Lib.SAMPLES_PER_SEC)
        if(vector[1]==1):
            signal=signal+Numpy.sin(2 * Numpy.pi * t * 2927 / Lib.SAMPLES_PER_SEC)
        if(vector[1]==0):
            signal=signal+Numpy.sin(2 * Numpy.pi * t * (-2927) / Lib.SAMPLES_PER_SEC)

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
        for i in range(int(Numpy.floor(record.size - (Lib.NUMBER_DATA_SAMPLES + Lib.NUMBER_NOISE_SAMPLES)))):
            dotProduct = Numpy.dot(noiseToSyncOn,
                                   record[i:Lib.NUMBER_NOISE_SAMPLES + i])
            if (dotProduct > maxDotProduct):
                maxDotProduct = dotProduct
                index = i

        begin = index + int(Lib.NUMBER_NOISE_SAMPLES)
        #begin=181815
        end = begin + int(Lib.NUMBER_DATA_SAMPLES)

        bla = record[begin:end]
        Plot.plot(1.5 * record)
        Plot.plot(Numpy.concatenate([Numpy.zeros(begin), bla, Numpy.zeros(end - begin)]))
        Plot.show()

        return bla

    def decodeur2LEspace(self,signal,nonoise):
        phaseSeeker=128
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        sinus = Numpy.zeros([phaseSeeker, len(t)])
        cosinus = Numpy.zeros([phaseSeeker, len(t)])

        if (nonoise == 1):
            f = Lib.LOWER_LOW_FREQUENCY_BOUND+Lib.FREQUENCY_STEP
        else:
            f = Lib.LOWER_UPPER_FREQUENCY_BOUND+Lib.FREQUENCY_STEP
        j=0
        for i in range (phaseSeeker):
            sinus[j]=Numpy.sin((2*Numpy.pi*t*2053/Lib.SAMPLES_PER_SEC)-1.5+i*0.05)
            cosinus[j]=Numpy.sin((2*Numpy.pi*t*2923/Lib.SAMPLES_PER_SEC)-1.5+i*0.05)
            j=j+1

        #plot signal versus sin
        #Plot.plot(0.1*Numpy.tile(sinus[0,:], [1,2])[0,:])
        #Plot.plot(signal[:2*Lib.ELEMENTS_PER_CHUNK])
        #Plot.plot(Numpy.zeros(2*Lib.ELEMENTS_PER_CHUNK))
        #Plot.show()
        print (signal.shape)
        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])
        dotArraySin=Numpy.zeros([chunks.shape[0],phaseSeeker])
        dotArrayCos=Numpy.zeros([chunks.shape[0],phaseSeeker])
        i=0
        resultArray=[]
        currphaseSin=0
        currphaseCos=30
        for chunk in chunks:
            dotArraySin[i,:]=chunk @ sinus.T
            dotArrayCos[i,:]=chunk @ cosinus.T
            #Plot.plot(dotArraySin[i,:])
            #Plot.show()
            #Plot.plot(dotArrayCos[i,:])
            #Plot.show()
            #!! And here starts the fun !!
            minSin=0
            maxSin=0
            jmaxSin=None
            jminSin=None
            minCos=0
            maxCos=0
            jmaxCos=None
            jminCos=None
            for j in range(phaseSeeker):
                if (dotArraySin[i][j]>maxSin):
                    maxSin=dotArraySin[i][j]
                    jmaxSin=j
                elif (dotArraySin[i][j]<minSin):
                    minSin=dotArraySin[i][j]
                    jminSin=j
                if (dotArrayCos[i][j]>maxCos):
                    maxCos=dotArrayCos[i][j]
                    jmaxCos=j
                elif (dotArrayCos[i][j]<minCos):
                    minCos=dotArrayCos[i][j]
                    jminCos=j
            jdistminSin=self.findClosestIndex(currphaseSin,jminSin,jmaxSin,phaseSeeker)
            jdistminCos=self.findClosestIndex(currphaseCos,jminCos,jmaxCos,phaseSeeker)
            vect=[]
            if (dotArraySin[i][jdistminSin]<0):
                vect.append(0)
                currphaseSin=jdistminSin
            else:
                vect.append(1)
                currphaseSin=jdistminSin
            if (dotArrayCos[i][jdistminCos]<0):
                vect.append(0)
                currphaseCos=jdistminCos
            else:
                vect.append(1)
                currphaseCos=jdistminCos
            print("currphaseSin",currphaseSin)
            print("currphaseCos",currphaseCos)
            print(vect)
            resultArray.append(vect)
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
