import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer
import numpy as Numpy
import matplotlib.pyplot as Plot

if __name__ == "__main__":
    # TODO: REVIEW

    io = IODeux.IODeux()
    coder = Coder.Coder()
    synthesizer = Synthesizer.Synthesizer()

    print("Detecting Noise")
    noNoise = synthesizer.detectNoise()

    print("Recording Signal")
    recording = synthesizer.recordSignal()
    # Numpy.save("recording_fourierv22", recording)
    # recording = Numpy.load("recording_fourierv22.npy")
    Plot.plot(recording)
    Plot.show()

    print("Extracting Data Signal")
    #dataSignal = synthesizer.extractDataSignal(recording)
    receivedVectors = synthesizer.decodeSignalToBitVectors(
        recording, noNoise).tolist()
    receivedVectors = list(
        map(lambda x: list(map(lambda y: int(y), x)), receivedVectors))
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

    stringRead = io.readFile(Lib.FILENAME_READ)
    encodedVectors = coder.encode(stringRead)
    flattenReal = Numpy.array(encodedVectors).flatten()
    flattenDecoded = Numpy.array(receivedVectors).flatten()
    zipped = zip(flattenReal.tolist(), flattenDecoded.tolist())
    compared = list(map(lambda x: 0 if x[0] == x[1] else 1, zipped))
    for i in range(len(compared)):
        if(compared[i] == 1):
            print(i)
    counted = Numpy.sum(Numpy.array(compared))

    print("ERRORS: ", counted)
