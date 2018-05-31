import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer
import sounddevice as SoundDevice
import numpy as Numpy

if __name__ == "__main__":
    io = IODeux.IODeux()
    coder = Coder.Coder()
    synthesizer = Synthesizer.Synthesizer()

    stringRead = io.readFile(Lib.FILENAME_READ)

    encodedVectors = coder.encode(stringRead)
    print("ENCODED VECTORS:")
    print(encodedVectors)

    noNoise = synthesizer.detectNoise()
    synchNoise = synthesizer.createWhiteNoise()

    print("Building Signal...")
    zeros = Numpy.zeros(Lib.SAMPLES_PER_SEC)

    tmp = synthesizer.generateCompleteSignal(encodedVectors[0:51], noNoise)
    #print("encodedVectors[0:(51)]")
    signalToSend = Numpy.concatenate([synchNoise, tmp])
    for i in range(1,40):
        tmp = synthesizer.generateCompleteSignal(encodedVectors[51*i:(51+51*i)], noNoise)
        #print(i," encodedVectors[",51*i,":(",51+51*i,")]")
        tmp2 = Numpy.concatenate([signalToSend, zeros])
        signalToSend = Numpy.concatenate([tmp2, tmp])
        #print(len(signalToSend))

    print(signalToSend)
    print("Done")
    SoundDevice.play(signalToSend)
    SoundDevice.wait()
