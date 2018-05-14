# Receive the k bit vectors from the coder and generate the corresponding sound/signal.
# Send in both noise-free bands at the same time. Each chunk sent during T seconds.
# + All reverse ops (from signal you listen to, to bit vector)...
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import lib as lib
import noise as noise

# Detect the noise type
def detectNoise():
    sd.default.channels = 1
    record = sd.rec(lib.FS, lib.FS, blocking=True)[
        :, 0]  # during 1s for now, can be reduced
    sd.wait()
    # record=np.load("NONOISE2.npy") --debug
    recordfft = np.fft.fft(record)
#    plt.plot(np.abs(recordfft))
#    plt.show()
    # f = np.fft.fftfreq(len(record), 1/lib.FS)  # OK ÇA SERT A QUOI ?
    sum1000 = np.sum(np.abs(recordfft[1000:2000]))
    sum2000 = np.sum(np.abs(recordfft[2000:3000]))
    if (sum1000 > sum2000):
        print("NONOISE=2")
        return 2
    else:
        print("NONOISE=1")
        return 1



# Create a white noise with a N(0,1), with the seed 42
def createWhiteNoise(time=lib.NOISE_TIME, seed=42):
    sample = int(lib.FS*time)
    np.random.seed(seed)
    noise = np.random.normal(0, 1, sample)
    return noise


def sendArrayVector(array,nonoise):
    signal = np.zeros(0)
    for a in array:
        inter = sendVectorInBases(a,nonoise)
        signal = np.concatenate([signal, inter])
    return signal



# Send vector in in basis 1=>1, 0=>-1, Basicly the same as sendVector
# but send -sin when 0 instead of 0
def sendVectorInBases(vector,nonoise,time=lib.TIME_BY_CHUNK,):
    k = lib.CHUNK_SIZE
    freqs = []
    # calculate the frequencies
    step = 1000/(k+1)
    if (nonoise == 1):
        for i in range(0, k):
            if(vector[i] == 1):
                freqs.append(1000+step*(i+1))
            else:
                freqs.append(-(1000+step*(i+1)))
    else:
        for i in range(0, k):
            if(vector[i] == 1):
                freqs.append(2000+step*(i+1))
            else:
                freqs.append(-(2000+step*(i+1)))
    # prepare the sinuses
    t = np.arange(time*lib.FS)
    signal = np.zeros(t.shape)
    print("Sending at frequencie(s) : ")
    for f in freqs:
        signal = signal + 10*np.sin(2*np.pi*t*f/lib.FS)  # 1st noise
        print(f)
    return signal



# time : in seconds
def receive(time=10*lib.TIME_BY_CHUNK+lib.NOISE_TIME):
    sd.default.channels = 1
    record = sd.rec(time*lib.FS, lib.FS, blocking=True)
    return record[:, 0]


# Synchronise the record, return the sub-array of record starting at the end of
# the white noise with the length TOTAL_ELEM_NUMBER
def sync(record, length):
    noise = createWhiteNoise()
    noiseLength = int(lib.FS*lib.NOISE_TIME)
    maxdot = 0
    index = 0
    # CHANGE TO TOTAL_ELEM_NUMBER
    for i in range(record.size - (lib.TIME_BY_CHUNK*lib.FS * length+noiseLength)):
        dot = np.dot(noise, record[i:noiseLength+i])
        if (dot > maxdot):
            maxdot = dot
            index = i
        i += 1
    begin = int(index+lib.NOISE_TIME*lib.FS)
    end = begin+lib.TIME_BY_CHUNK*lib.FS * length  # CHANGE TO TOTAL TIME
    print(begin,end)
    return record[begin:end]



# Make the dot product with the basis to detect the codeword
def projectOnBasis(signal,nonoise):
    # calculate the basis
    k = lib.CHUNK_SIZE
    step = 1000/(k+1)
    t = np.arange(lib.TIME_BY_CHUNK*lib.FS)
    sinus = np.zeros([lib.CHUNK_SIZE, len(t)])
    for i in range(0, k):
        if (nonoise == 1):
            f = 1000+step*(i+1)
        else:
            f = 2000+step*(i+1)
        sinus[i, :] = np.sin(2*np.pi*t*f/lib.FS)
    # Make the projection
    i = 0
    resultArray = []
    for s in sinus:
        dot = np.dot(s, signal)
        resultArray.append(1 if (dot >= 0) else 0)
        print(i, dot)
        i = i + 1

    print("Result Array:")
    print(resultArray)
    return resultArray


# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def decodeSignal(signal,nonoise):
    chunks = [signal[i:i + lib.ELEMENTS_PER_CHUNK]
              for i in range(0, len(signal), lib.ELEMENTS_PER_CHUNK)]
    i = 0
    for chunk in chunks:
        print(len(chunk))
        chunks[i] = projectOnBasis(chunk,nonoise)
        i += 1
    print("Chunk Array:")
    print(chunks)
    return chunks




# TEST
# Sending
nonoise=detectNoise()
noise1=createWhiteNoise(lib.NOISE_TIME)
a = [[0],[1],[1],[0],[0],[1],[0],[0],[0],[1],[1],[0],[0],[1],[0],[0],
     [0],[1],[1],[0],[0],[1],[0],[0],[0],[1],[1],[0],[0],[1],[0],[0]]
signal=sendArrayVector(a,nonoise)
fullSignal=np.concatenate([noise1,signal])
sd.play(fullSignal)
sd.wait()


# Local test
"""
noise1 = createWhiteNoise()
noise2 = createWhiteNoise(lib.NOISE_TIME, 3)

a = [[1, 0], [1, 1], [0, 0]]
length = len(a)

NONOISE = 2
if(NONOISE == 2):
    noise3 = noise.band_limited_noise(
        2000, 3000, lib.FS*lib.TIME_BY_CHUNK * length, lib.FS)*100000
else:
    noise3 = noise.band_limited_noise(
        1000, 2000, lib.FS*lib.TIME_BY_CHUNK * length, lib.FS)*100000
signal = sendArrayVector(a,NONOISE)
signal = signal+noise3
signal += noise3
midSignal = np.concatenate([noise1, signal])
fullSignal = np.concatenate([noise2, midSignal])
# plt.plot(fullSignal)
# plt.show()
sync = sync(fullSignal, length)
# plt.plot(sync)
# plt.show()
# peaks=findPeaks(sync,1)
decodeSignal(signal,NONOISE)
"""

#Receiving
"""
nonoise=detectNoise()
rec=receive()
sync=sync(rec,3)
plt.plot(sync)
plt.show()
nonoise=1
decodeSignal(sync,nonoise)
"""


# tests with sync.numpy
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
