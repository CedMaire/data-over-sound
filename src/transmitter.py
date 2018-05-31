import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer
import sounddevice as SoundDevice
import numpy as Numpy
import matplotlib.pyplot as Plot

if __name__ == "__main__":
    # TODO: REVIEW

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
    signalToSend = synthesizer.generateCompleteSignal(encodedVectors, noNoise)
    signalToSend = Numpy.concatenate([synchNoise, signalToSend])
    Plot.plot(signalToSend)
    Plot.show()
    #SoundDevice.play(signalToSend, Lib.SAMPLES_PER_SEC, blocking=True)
