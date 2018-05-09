# Receive the k bit vectors from the coder and generate the corresponding sound/signal.
# Send in both noise-free bands at the same time. Each chunk sent during T seconds.
# + All reverse ops (from signal you listen to, to bit vector)...
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import lib as lib




#Create a white noise with a N(0,1), with the seed 42
def createWhiteNoise(time=lib.NOISE_TIME,seed=42):
    sample=lib.FS*time
    np.random.seed(seed)
    noise=np.random.normal(0,1,sample)
    return noise


# send an array, k bit at a time. It will devide the frequency domain in equal
# part and send at the divinding frequencies. Produce for the two domain.
# use time to change in sec the time of the transmission
#return the sended array
def sendBitArray(array, time=lib.TIME_BY_CHUNK):
    k = len(array)
    freqs = []
    # calculate the frequencies
    step = 1000/(k+1)
    for i in range(0, k):
        if(array[i] == 1):
            freqs.append(1000+step*(i+1))
    # prepare the sinuses
    t = np.arange(time*lib.FS)
    signal = np.zeros(t.shape)
    print("Sending at frequencie(s) (also sending at f+1000) : ")
    for f in freqs:
        signal = signal + np.sin(2*np.pi*t*f/lib.FS)  # 1st noise
        signal = signal + np.sin(2*np.pi*t*(1000+f)/lib.FS)  # 2nd noise
        print(f)
    return signal

#time : in seconds
def receive(time=2*lib.TIME_BY_CHUNK):
    sd.default.channels=1
    record=sd.rec(time*lib.FS,lib.FS,blocking=True)
    return record[:,0]

#Synchronise the record, return the sub-array of record starting at the end of
#the white noise with the length TOTAL_ELEM_NUMBER
def sync(record):
    noise=createWhiteNoise()
    noiseLength=lib.FS*lib.NOISE_TIME
    maxdot=0
    index=0
    for i in range (record.size - noiseLength):
        dot=np.dot(noise,record[i:noiseLength+i])
        dotarray[i]=dot
        if (dot>40): print(i,dot)
        if (dot< -40): print(i,dot)

        if (dot>maxdot):
            maxdot=dot
            index=i
        i+=1
    begin=index+lib.NOISE_TIME*lib.FS
    end = begin+lib.FS ###CHANGE TO TOTAL TIME
    return record[begin:end]

def findPeaks(signal, ones,frequence=lib.FS):
    w = np.fft.fft(signal)
    f = np.fft.fftfreq(len(w),d=1/frequence)
    peaks = np.empty(2*ones)
    i = 0
    for x in range(2*ones):
        idx = np.argmax(np.abs(w))
        freq_in_hertz = f[idx]
        peaks[i]=freq_in_hertz

        w = np.delete(w, idx)
        #idx = np.argmax(np.abs(w))
        #w = np.delete(w, idx)
        i+=1
    peaks=np.sort(peaks)
    return peaks



#TEST

#Sending
'''
noise=createWhiteNoise(lib.NOISE_TIME)
a = [1]
signal=sendBitArray(a)
fullSignal=np.concatenate([noise,signal])
plt.plot(fullSignal)
plt.show()
sd.play(fullSignal)
sd.wait()
'''

#Local test
'''
noise=createWhiteNoise()
noise2=createWhiteNoise(lib.NOISE_TIME,3)
noise3=createWhiteNoise(lib.TIME_BY_CHUNK,1)
a = [1]
signal=sendBitArray(a)
signal+=noise3
midSignal=np.concatenate([noise,signal])
fullSignal=np.concatenate([noise2,midSignal])
plt.plot(fullSignal)
plt.show()
sync=sync(fullSignal)
plt.plot(sync)
plt.show()
peaks=findPeaks(sync,1)
print(peaks)
'''

#Receiving
'''
rec=receive()
print(rec.shape)
#np.save("sync",rec)
plt.plot(rec)
plt.show()
sync=sync(rec)
#np.save("sync",sync)
plt.plot(sync)
plt.show()
peaks=findPeaks(sync,10)
print(peaks)
'''

#tests with sync.numpy

rec=np.load("rec.npy")[:,0]
sinus=np.load("sinus1500.npy")
sinus=sinus[0:lib.FS]
plt.plot(rec)
plt.show()
sync=sync(rec)
plt.plot(sync)
plt.plot(sinus)
plt.show()
findPeaks(sync,10)
