# Main receiver program.
import numpy as np
import sounddevice as sd

#time : in seconds
def receive(time):
    fs=44100
    sd.default.channels=1
    record=sd.rec(10*fs,fs,blocking=True) #See if we can't do this in background
    print(record)

receive(10)
