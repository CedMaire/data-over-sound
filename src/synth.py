import numpy as Numpy
import sounddevice as SoundDevice
import matplotlib.pyplot as Plot
import iodeux as IODeux
import lib as Lib
import coder as Coder
import numpy as np
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
        # numberofbits = 8
        # sig00 = self.generateVectorSignal(Numpy.array([0]), nonoise)
        # sins=  Numpy.zeros([2*numberofbits, len(sig00)])
        # sins[0,:]=sig00
        #
        # for i in range(1, numberofbits):
        #     sins[i,:]=self.generateVectorSignal(Numpy.array([i]), nonoise)
        sig00 = self.generateVectorSignal(Numpy.array([0]), nonoise)
        sig01 = self.generateVectorSignal(Numpy.array([1]), nonoise)
        sins = Numpy.zeros([2,len(sig00)])
        sins[0,:]=sig00
        sins[1,:]=sig01# c'est moche mais c'est pour matcher avec votre truc
        array = Numpy.array(array)
        return sins[array,:].reshape([-1]) #va falloir hardcoder pour augmenter la dimension...


    def generateVectorSignal(self, vector, nonoise):
        f = np.copy(Lib.f1) if nonoise==1 else np.copy(Lib.f2)
        for i in range(len(vector)):
            if (vector[i]==0):
                f[i]=-f[i]
        t = np.arange(Lib.ELEMENTS_PER_CHUNK)
        sin = np.sum(np.sin(2 * np.pi * t[np.newaxis,:] * f[:,np.newaxis] / Lib.SAMPLES_PER_SEC), axis=0)
        return sin

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

    def decodeur2LEspace(self,signal,f):
        phaseSeeker=128
        t = Numpy.arange(Lib.ELEMENTS_PER_CHUNK)
        sinus = Numpy.zeros([phaseSeeker, len(t)])
        for i in range (phaseSeeker):
            sinus[i]=Numpy.sin((2*Numpy.pi*t*f/Lib.SAMPLES_PER_SEC)-1.5+i*0.05)
        chunks = signal.reshape([-1, Lib.ELEMENTS_PER_CHUNK])
        dotArray=Numpy.zeros([chunks.shape[0],phaseSeeker])
        i=0
        resultArray=np.zeros(int(Lib.NEEDED_AMOUNT_OF_VECTORS/Lib.CHUNK_SIZE))
        currphase=0
        for chunk in chunks:
            dotArray[i,:]=chunk @ sinus.T
            #!! And here starts the fun !!
            min=0
            max=0
            jmax=None
            jmin=None
            for j in range(phaseSeeker):
                if (dotArray[i][j]>max):
                    max=dotArray[i][j]
                    jmax=j
                elif (dotArray[i][j]<min):
                    min=dotArray[i][j]
                    jmin=j
            jdistmin=self.findClosestIndex(currphase,jmin,jmax,phaseSeeker)
            if (dotArray[i][jdistmin]<0):
                resultArray[i]=(0)
            else:
                resultArray[i]=(1)
            currphase=jdistmin
            i=i+1

        print("resultArray",resultArray)
        print("len(resultArray)",len(resultArray))
        return resultArray

    def decodeAllFreqs(self,signal,nonoise):
        f= np.copy(Lib.f1) if nonoise==1 else np.copy(Lib.f2)
        results=np.zeros((Lib.CHUNK_SIZE, Lib.NEEDED_AMOUNT_OF_VECTORS))
        print((Lib.NEEDED_AMOUNT_OF_VECTORS))
        for i in range(Lib.CHUNK_SIZE):
            print(i)
            results[i,:]=self.decodeur2LEspace(signal,f[i])
        #return r_[     ]
        #results = np.hstack(results[0], results[1])
        print("results.shape1", results.shape)
        results=list(map(lambda x : list(map(lambda y: int(y),x )), results.reshape(len(results[0]),1).tolist()))
        #print("results.shape2", results.shape)

        # for i in range(len(results)):
        #     results[i] = np.array([results[i]])
        print("results", results)
        # np.d
        return results


    def findClosestIndex(self,j0,j1,j2,phaseSeeker):
        d1=Numpy.abs(j1-j0)
        if(d1>phaseSeeker/2):
            d1=phaseSeeker-d1
        d2=Numpy.abs(j2-j0)
        if(d2>phaseSeeker/2):
            d2=phaseSeeker-d2
        #print("debug", "d1",d1,"j1",j1,"d2",d2,"j2",j2)
        return j1 if d1<d2 else j2
