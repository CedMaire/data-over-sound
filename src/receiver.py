import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer
import numpy as Numpy
import matplotlib.pyplot as Plot
import numpy as np

if __name__ == "__main__":
    io = IODeux.IODeux()
    coder = Coder.Coder()
    synthesizer = Synthesizer.Synthesizer()
    stringRead = io.readFile(Lib.FILENAME_READ)
    encodedVectors = coder.encode(stringRead)

    print("Detecting Noise")
    noNoise = synthesizer.detectNoise()
    print("Recording Signal")
    recording = synthesizer.recordSignal()
    #Numpy.save("recording_pierre", recording)
    #recording = Numpy.load("recording_pierre.npy")
    Plot.plot(recording)
    Plot.show()
    print("Extracting Data Signal")
    dataSignal = synthesizer.extractDataSignal(recording)
    #receivedVectors = synthesizer.decodeSignalToBitVector(dataSignal, noNoise)
    receivedVectors_vec = [synthesizer.decodeur2LEspace(dataSignal, noNoise) , synthesizer.decodeur3LEspace(dataSignal, noNoise), synthesizer.decodeur3LEspace(dataSignal, noNoise, True)]


    for receivedVectors in receivedVectors_vec:
        l = min(len(receivedVectors),len(encodedVectors))
        L = max(len(receivedVectors),len(encodedVectors))
        print(l,L)
        diff = np.sum(np.array(receivedVectors)[:l]!=np.array(encodedVectors)[:l])+(L-l)
        print(diff)

        Plot.plot(np.logical_xor(receivedVectors[:l], encodedVectors[:l]).reshape([-1]))
        Plot.show()


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
