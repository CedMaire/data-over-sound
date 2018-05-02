import numpy as np
import sounddevice as sd

def createRandomArray(array_size, seed, max_value):
	np.random.seed(seed)
	tmp = np.random.randint(max_value, size=(array_size))
	return tmp


print(createRandomArray(1000,8,2))

def sendBitArray(array, time):
    k=len(array)
    fs = 44100
    freqs=[]
    #calculate the frequencies
    step=1000/(k+1)
    for i in range (0,k):
        if(array[i]==1):
            freqs.append(1000+step*(i+1))
    #prepare the sinuses
    t=np.arange(time*fs)
    signal=np.zeros(t.shape)
    print("Sending at base frequencie(s) (also sending at f+1000) : ")
    for f in freqs:
        signal=signal + np.sin(2*np.pi*t*f/fs) #1st noise
        signal=signal + np.sin(2*np.pi*t*(1000+f)/fs) #2nd noise
        print(f)
    sd.play(signal,fs)
    sd.wait()

#sendBitArray(createRandomArray(50, 48, 2), 10)