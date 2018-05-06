# Receive the k bit vectors from the coder and generate the corresponding sound/signal.
# Send in both noise-free bands at the same time. Each chunk sent during T seconds.
# + All reverse ops (from signal you listen to, to bit vector)...
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

fs = 44100
# send an array, k bit at a time. It will devide the frequency domain in equal
# part and send at the divinding frequencies. Produce for the two domain.
# use time to change in sec the time of the transmission
def sendBitArray(array, time):
    k = len(array)
    freqs = []
    # calculate the frequencies
    step = 1000/(k+1)
    for i in range(0, k):
        if(array[i] == 1):
            freqs.append(1000+step*(i+1))
    # prepare the sinuses
    t = np.arange(time*fs)
    signal = np.zeros(t.shape)
    print("Sending at frequencie(s) (also sending at f+1000) : ")
    for f in freqs:
        signal = signal + np.sin(2*np.pi*t*f/fs)  # 1st noise
        signal = signal + np.sin(2*np.pi*t*(1000+f)/fs)  # 2nd noise
        print(f)
    #sd.play(signal, fs)
    x=np.arange(0,500,1)
    sub=signal[0:500]
    plt.plot(x, sub)
    plt.show()

#TEST

#TEST
    #sd.wait()

#time : in seconds
def receiveAndFFT(time):
    sd.default.channels=1
    record=sd.rec(time*fs,fs,blocking=True)
    rfft=np.fft.rfft(signal[0:fs])
    hz=np.arange(0,fs//2+1)
    plt.plot(hz,rfft)
    plt.show()



#TEST
a = [0, 0, 1, 0, 0] #apparently, 1500hz doesn't work...
sendBitArray(a, 2)
#receiveAndFFT(2)
