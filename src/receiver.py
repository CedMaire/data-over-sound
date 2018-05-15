import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer

if __name__ == "__main__":
    io = IODeux.IODeux()
    coder = Coder.Coder()
    synthesizer = Synthesizer.Synthesizer()

    noNoise = synthesizer.detectNoise()
    recording = synthesizer.recordSignal()
    dataSignal = synthesizer.extractDataSignal(recording)
    encodedVectors = synthesizer.decodeSignalToBitVectors(dataSignal, noNoise)

    decodedTuple = coder.decode(encodedVectors)
    decodedString = decodedTuple[1]
    print("DECODED STRING:")
    print(decodedString)

    if (decodedTuple[0]):
        io.writeFile(Lib.FILENAME_WRITE, decodedString)
        print("Done!")
    else:
        # Try to flip the bits?
        print(decodedTuple[1])
