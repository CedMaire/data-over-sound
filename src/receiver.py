import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer
import numpy as Numpy

if __name__ == "__main__":
    io = IODeux.IODeux()
    coder = Coder.Coder()
    synthesizer = Synthesizer.Synthesizer()
    print("Detecting Noise")
    noNoise = synthesizer.detectNoise()
    print("Recording Signal")
    recording = synthesizer.recordSignal()
    print("Extracting Data Signal")
    dataSignal, end = synthesizer.extractDataSignal(recording)
    for i in range(39):
        temp = synthesizer.extractDataSignal(recording[end:len(recording)])
        dataSignal = Numpy.concatenate([dataSignal, temp])
    receivedVectors = synthesizer.decodeSignalToBitVectors(dataSignal, noNoise)
    print(receivedVectors)

    decodedTuple = coder.decode(receivedVectors)
    decodedString = decodedTuple[1]
    print("DECODED STRING:")
    print(decodedString)

    if (decodedTuple[0]):
        io.writeFile(Lib.FILENAME_WRITE, decodedString)
        print("Done!")
    else:
        # Try to flip the bits?
        print(decodedTuple[1])
