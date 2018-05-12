# Receive the k bit vectors from the coder and generate the corresponding sound/signal.
# Send in both noise-free bands at the same time. Each chunk sent during T seconds.
# + All reverse ops (from signal you listen to, to bit vector)...
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import lib as lib
import noise as noise




#Create a white noise with a N(0,1), with the seed 42
def createWhiteNoise(time=lib.NOISE_TIME,seed=42):
    sample=lib.FS*time
    np.random.seed(seed)
    noise=np.random.normal(0,1,sample)
    return noise

def sendArrayVector(array):
  signal=np.zeros(0)
  for a in array:
      #inter=sendVector(a)
      inter=sendVectorInBases(a)
      signal=np.concatenate([signal,inter])
  return signal

# send an vector, k bit at a time. It will devide the frequency domain in equal
# part and send at the divinding frequencies. Produce for the two domain.
# use time to change in sec the time of the transmission
#return the sended array
def sendVector(vector, time=lib.TIME_BY_CHUNK):
    k = len(vector)
    freqs = []
    # calculate the frequencies
    step = 1000/(k+1)
    for i in range(0, k):
        if(vector[i] == 1):
            freqs.append(1000+step*(i+1))
    # prepare the sinuses
    t = np.arange(time*lib.FS)
    signal = np.zeros(t.shape)
    print("Sending at frequencie(s): ")
    for f in freqs:
        signal = signal + np.sin(2*np.pi*t*f/lib.FS)  # 1st noise
        signal = signal + np.sin(2*np.pi*t*(1000+f)/lib.FS)  # 2nd noise
        print(f,f+1000)
    return signal

#Send vector in in basis 1=>1, 0=>-1, Basicly the same as sendVector
# but send -sin when 0 instead of 0
def sendVectorInBases(vector, time=lib.TIME_BY_CHUNK):
    k = lib.CHUNK_SIZE
    freqs = []
    # calculate the frequencies
    step = 1000/(k+1)
    for i in range(0, k):
        if(vector[i] == 1):
            freqs.append(1000+step*(i+1))
        else:
            freqs.append(-(1000+step*(i+1)))
    # prepare the sinuses
    t = np.arange(time*lib.FS)
    signal = np.zeros(t.shape)
    print("Sending at frequencie(s) : ")
    for f in freqs:
        signal = signal + np.sin(2*np.pi*t*f/lib.FS)  # 1st noise
        if (f>0):
            signal = signal + np.sin(2*np.pi*t*(1000+f)/lib.FS)  # 2nd noise
            print(f,f+1000)
        else :
            signal = signal + np.sin(2*np.pi*t*(-1000+f)/lib.FS)  # 2nd noise
            print(f,f-1000)
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
    for i in range (record.size - lib.TIME_BY_CHUNK*lib.FS - noiseLength): ### CHANGE TO TOTAL_ELEM_NUMBER
        dot=np.dot(noise,record[i:noiseLength+i])
        if (dot>maxdot):
            maxdot=dot
            index=i
        i+=1
    begin=index+lib.NOISE_TIME*lib.FS
    end = begin+lib.TIME_BY_CHUNK*lib.FS ###CHANGE TO TOTAL TIME
    return record[begin:end]

def findPeaks(signal, ones,frequence=lib.FS):
    w = np.fft.fft(signal)
    plt.plot(np.abs(w))
    plt.show()
    f = np.fft.fftfreq(len(w))
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

#Make the dot product with the basis to detect the codeword
def projectOnBasis(signal):
    #calculate the basis
    k = lib.CHUNK_SIZE
    freqs = []
    step = 1000/(k+1)
    t = np.arange(lib.TIME_BY_CHUNK*lib.FS)
    print(len(t))
    sinus= np.zeros([lib.CHUNK_SIZE*2,len(t)])
    for i in range(0, k):
        print(i)
        f=1000+step*(i+1)
        sinus[2*i,:]=np.sin(2*np.pi*t*f/lib.FS)
        sinus[2*i+1,:]=np.sin(2*np.pi*t*(1000+f)/lib.FS)
        print(f,1000+f)
        i=i+1
    #Make the projection
    i=0
    for s in sinus:
        plt.plot(s)
        plt.show()
        dot=np.dot(s,signal)
        print(i,dot)
        i=i+1
#TEST

#Sending
'''
noise=createWhiteNoise(lib.NOISE_TIME)
a = [1,1]
signal=sendBitArray(a)
fullSignal=np.concatenate([noise,signal])
plt.plot(fullSignal)
plt.show()
sd.play(fullSignal)
sd.wait()
'''

#Local test

noise1=createWhiteNoise()
noise2=createWhiteNoise(lib.NOISE_TIME,3)
noise3=noise.band_limited_noise(1000,2000,lib.FS*lib.TIME_BY_CHUNK,lib.FS)*1000000
a = [[1,1]]
signal=sendArrayVector(a)
plt.plot(signal)
plt.show()
signal+=noise3
midSignal=np.concatenate([noise1,signal])
fullSignal=np.concatenate([noise2,midSignal])
#plt.plot(fullSignal)
#plt.show()
sync=sync(fullSignal)
#plt.plot(sync)
#plt.show()
#peaks=findPeaks(sync,1)
projectOnBasis(signal)


#Receiving
'''
rec=receive()
print(rec.shape)
#np.save("sync",rec)
sync=sync(rec)
#np.save("sync",sync)
plt.plot(sync)
plt.show()
peaks=findPeaks(sync,10)
print(peaks)
'''

#tests with sync.numpy
'''
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
'''
