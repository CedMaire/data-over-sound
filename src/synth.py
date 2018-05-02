# Receive the k bit vectors from the coder and generate the corresponding sound/signal.
# Send in both noise-free bands at the same time. Each chunk sent during T seconds.
# + All reverse ops (from signal you listen to, to bit vector)...
import numpy as np
import sounddevice as sd


# send an array, k bit at a time. It will devide the frequency domain in equal
# part and send at the divinding frequencies. Produce for the two domain.
# use time to change in sec the time of the transmission
def sendBitArray(array, time):
    k = len(array)
    fs = 44100
    freqs = []
    # calculate the frequencies
    step = 1000/(k+1)
    for i in range(0, k):
        if(array[i] == 1):
            freqs.append(1000+step*(i+1))
    # prepare the sinuses
    t = np.arange(time*fs)
    signal = np.zeros(t.shape)
    print("Sending at base frequencie(s) (also sending at f+1000) : ")
    for f in freqs:
        signal = signal + np.sin(2*np.pi*t*f/fs)  # 1st noise
        signal = signal + np.sin(2*np.pi*t*(1000+f)/fs)  # 2nd noise
        print(f)
    sd.play(signal, fs)
    sd.wait()


# Test
a = [1, 0, 1, 0, 1]
sendBitArray(a, 10)
