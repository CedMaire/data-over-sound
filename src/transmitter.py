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

    encodedVectors = encodedVectors[0:int(len(encodedVectors) / 51)]

    noNoise = synthesizer.detectNoise()
    synchNoise = synthesizer.createWhiteNoise()

    print("Building Signal...")
    signalToSend = synthesizer.generateCompleteSignal(encodedVectors, noNoise)
    signalToSend = Numpy.concatenate([synchNoise, signalToSend])

    SoundDevice.play(signalToSend,6000)
    SoundDevice.wait()
