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

    # stringRead = io.readFile(Lib.FILENAME_READ)

    # encodedVectors = coder.encode(stringRead)

    print("Detecting Noise")
    noNoise = synthesizer.detectNoise()
    #noNoise = 1
    print("Recording Signal")
    recording = synthesizer.recordSignal()
    #Numpy.save("rec_total_003__2-3_44k", recording)
    #recording = Numpy.load("rec_total_003__1-2_44k.npy")
    # Plot.plot(recording)
    # Plot.show()
    print("Extracting Data Signal")
    dataSignal = synthesizer.extractDataSignal(recording)
    #receivedVectors = synthesizer.decodeSignalToBitVector(dataSignal, noNoise)
    receivedVectors = synthesizer.decodeur2LEspace(dataSignal, noNoise)
    print(receivedVectors)

    decodedTuple = coder.decode(receivedVectors)
    decodedString = decodedTuple[1]
    print("DECODED STRING:")
    print(decodedString)

    # flattenReal = Numpy.array(encodedVectors).flatten()
    # flattenDecoded = Numpy.array(receivedVectors).flatten()
    # zipped = zip(flattenReal.tolist(), flattenDecoded.tolist())
    # compared = list(map(lambda x: 0 if x[0] == x[1] else 1, zipped))
    # counted = Numpy.sum(Numpy.array(compared))

    # print("ERRORS: ", counted)

    if (decodedTuple[0]):
        io.writeFile(Lib.FILENAME_WRITE, decodedString)
        print("Done!")
    else:
        print(decodedTuple[1])
