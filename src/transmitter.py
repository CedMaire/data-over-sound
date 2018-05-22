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
    encodedVectors=encodedVectors[0:40]
    print("ENCODED VECTORS:")
    print(encodedVectors)

    noNoise = synthesizer.detectNoise()
    synchNoise = synthesizer.createWhiteNoise()

    print("Building Signal...")
    signalToSend = synthesizer.generateCompleteSignal(encodedVectors, noNoise)
    signalToSend = Numpy.concatenate([synchNoise, Numpy.zeros(6000),signalToSend])

    SoundDevice.play(signalToSend, Lib.SAMPLES_PER_SEC, blocking=True)
