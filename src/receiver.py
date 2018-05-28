import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer
import numpy as Numpy
import matplotlib.pyplot as Plot
if __name__ == "__main__":
    io = IODeux.IODeux()
    coder = Coder.Coder()
    synthesizer = Synthesizer.Synthesizer()

    print("Detecting Noise")
    noNoise = synthesizer.detectNoise()
    print("Recording Signal")
    recording = synthesizer.recordSignal()
    #Numpy.save("recording_80_22k", recording)
    #recording = Numpy.load("recording44.npy")
    #Plot.plot(recording)
    #Plot.show()
    print("Extracting Data Signal")
    dataSignal = synthesizer.extractDataSignal(recording)
    #receivedVectors = synthesizer.decodeSignalToBitVector(dataSignal, noNoise)
    receivedVectors = synthesizer.decodeAllFreqs(dataSignal, noNoise)
    print(receivedVectors)

    decodedTuple = coder.decode(receivedVectors)
    decodedString = decodedTuple[1]
    print("DECODED STRING:")
    print(decodedString)

    if (decodedTuple[0]):
        io.writeFile(Lib.FILENAME_WRITE, decodedString)
        print("Done!")
    else:
        print(decodedTuple[1])
