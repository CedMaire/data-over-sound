# Receive the k bit vectors from the coder and generate the corresponding sound/signal.
# Send in both noise-free bands at the same time. Each chunk sent during T seconds.
# + All reverse ops (from signal you listen to, to bit vector)...
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

fs = 44100

#Create and send a white noise with a N(0,1)
def sendWhiteNoise(time):
    sample=fs*time
    noise=np.random.normal(0,1,sample)
    sd.play(noise)
    sd.wait()
    #plt.plot(noise)
    #plt.show()

def findPeaks(frequence, signal, ones):
    w = np.fft.fft(signal)
    f = np.fft.fftfreq(len(w))
    #print(f)
    #print(f.min(), f.max())

    peaks = np.empty(2*ones)
    i = 0
    for x in range(2*ones):
        idx = np.argmax(np.abs(w))
        freq = f[idx]
        freq_in_hertz = abs(freq * frequence)
        print(freq_in_hertz)
        peaks[i]=freq_in_hertz

        w = np.delete(w, idx)
        idx = np.argmax(np.abs(w))
        w = np.delete(w, idx)
        i+=1

    print(peaks)    


# send an array, k bit at a time. It will devide the frequency domain in equal
# part and send at the divinding frequencies. Produce for the two domain.
# use time to change in sec the time of the transmission
def sendBitArray(array, time):
    k = len(array)
    freqs = []
    ones = 0
    # calculate the frequencies
    step = 1000/(k+1)
    for i in range(0, k):
        if(array[i] == 1):
            freqs.append(1000+step*(i+1))
            ones += 1
    # prepare the sinuses
    t = np.arange(time*fs)
    signal = np.zeros(t.shape)
    print("Sending at frequencie(s) (also sending at f+1000) : ")
    for f in freqs:
        signal = signal + np.sin(2*np.pi*t*f/fs)  # 1st noise
        signal = signal + np.sin(2*np.pi*t*(1000+f)/fs)  # 2nd noise
        print(f)
    #sd.play(signal, fs)
    #plotting the signal
    x=np.arange(0,500,1)
    sub=signal[0:500]
    #plt.plot(x, sub)
    #plt.show()

#TEST
    
    findPeaks(fs, signal, ones)
    
    # fft=np.fft.fft(signal[0:fs])
    # hz=np.arange(0,fs)
    # tmp_freq = np.fft.fftfreq(len(fft))
    # print(len(tmp_freq))
    # print(tmp_freq)
    # plt.plot(hz, np.abs(tmp_freq))
    # plt.plot(hz,np.abs(fft))
    # plt.show()

#/TEST
    #sd.wait()

#time : in seconds
def receiveAndFFT(time):
    sd.default.channels=1
    record=sd.rec(time*fs,fs,blocking=True)
    fft=np.fft.fft(record[0:fs])
    hz=np.arange(0,fs)
    plt.plot(hz,np.abs(fft))
    plt.show()



#TEST
#sendWhiteNoise(5)
a = [1, 0, 1, 0, 0] #apparently, 1500hz doesn't work...
sendBitArray(a, 5)
#receiveAndFFT(2)



    