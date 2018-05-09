# Receive the k bit vectors from the coder and generate the corresponding sound/signal.
# Send in both noise-free bands at the same time. Each chunk sent during T seconds.
# + All reverse ops (from signal you listen to, to bit vector)...
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import lib as lib
import scipy
from scipy import signal


# Create a white noise with a N(0,1), with the seed 42
def createWhiteNoise(time=lib.NOISE_TIME):
    sample = lib.FS*time
    np.random.seed(42)
    noise = np.random.normal(0, 1, sample)
    return noise


# send an array, k bit at a time. It will devide the frequency domain in equal
# part and send at the divinding frequencies. Produce for the two domain.
# use time to change in sec the time of the transmission
# return the sended array
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
    # sd.play(signal, fs)
    #
    return signal
# /TEST
    # sd.wait()

# time : in seconds


def receive(time=2*lib.TOTAL_ELEM_NUMBER):
    sd.default.channels = 1
    record = sd.rec(time*lib.FS, lib.FS, blocking=True)
    return record

# Synchronise the record, return the sub-array of record starting at the end of
# the white noise with the length TOTAL_ELEM_NUMBER


def sync(record):
    noise = createWhiteNoise()
    noiseLength = lib.FS*lib.NOISE_TIME
    maxdot = 0
    index = 0
    for i in range(record.size - noiseLength):
        dot = np.dot(noise, record[i:noiseLength+i])
        if (dot > maxdot):
            maxdot = dot
            index = i
        i += 1
    begin = index+lib.NOISE_TIME*lib.FS
    end = begin+lib.TOTAL_ELEM_NUMBER
    return record[begin:end]


def findPeaks(frequence, signal, ones):
    w = np.fft.fft(signal)
    f = np.fft.fftfreq(len(w))
    # plt.plot(w)
    # plt.show()
    # print(f)
    # print(f.min(), f.max())

    peaks = np.empty(2*ones)
    i = 0
    for _ in range(2*ones):
        idx = np.argmax(np.abs(w))
        freq = f[idx]
        freq_in_hertz = abs(freq * frequence)
        peaks[i] = freq_in_hertz

        w = np.delete(w, idx)
        #idx = np.argmax(np.abs(w))
        #w = np.delete(w, idx)
        i += 1
    peaks = np.sort(peaks)
    return peaks



# TEST
rec = receive(2 * lib.TIME_BY_CHUNK)

sync = sync(rec)
plt.plot(sync)
plt.show()

b, a = signal.butter(
    1, [1000 / (0.5 * lib.FS), 3000 / (0.5 * lib.FS)], "bandpass")
bandpassed = signal.filtfilt(b, a, sync)

plt.plot(bandpassed)
plt.show()

peaks = findPeaks(lib.FS, bandpassed, 10)
print(peaks)

'''
noise=createWhiteNoise(lib.NOISE_TIME)
a = [0, 0, 1, 0, 0]
signal=sendBitArray(a)
print(noise.shape, signal.shape)
sent=np.concatenate([noise,signal])
plt.plot(sent)
plt.show()
sync=sync(sent)
plt.plot(sync)
plt.show()
receiveAndFFT(2)
'''
