import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer
import noisedeux as Noise
import numpy as Numpy
import matplotlib.pyplot as Plot

if __name__ == "__main__":
    # TODO: REVIEW

    io = IODeux.IODeux()
    coder = Coder.Coder()
    synthesizer = Synthesizer.Synthesizer()
    noise = Noise.NoiseGenerator()

    stringRead = io.readFile(Lib.FILENAME_READ)

    # Encode
    encodedVectors = coder.encode(stringRead)
    print("ENCODED VECTORS:")
    print(encodedVectors)

    # Generate Signal
    #noiseStart = synthesizer.createWhiteNoise()
    #noiseEnd = synthesizer.createWhiteNoise()

    noNoise = 2  # CHANGE AS YOU WANT BETWEEN {1, 2}
    noiseMiddle = None
    if(noNoise == 2):
        noiseMiddle = noise.generateBandLimitedNoise(
            Lib.UPPER_LOW_FREQUENCY_BOUND,
            Lib.UPPER_UPPER_FREQUENCY_BOUND,
            Lib.NUMBER_DATA_SAMPLES,
            Lib.SAMPLES_PER_SEC) * 100
    else:
        noiseMiddle = noise.generateBandLimitedNoise(
            Lib.LOWER_LOW_FREQUENCY_BOUND,
            Lib.LOWER_UPPER_FREQUENCY_BOUND,
            Lib.NUMBER_DATA_SAMPLES,
            Lib.SAMPLES_PER_SEC) * 100

    signalToSend = synthesizer.generateCompleteSignal(encodedVectors, noNoise)

    # Send
    #signalToSend = Numpy.concatenate(
    #    [noiseStart, signalToSend])

    # Receive
    #dataSignal = synthesizer.extractDataSignal(signalToSend)
    receivedVectors = synthesizer.decodeSignalToBitVectors(signalToSend, noNoise)
    receivedVectors=list(map(lambda x : list(map(lambda y: int(y), x)),receivedVectors))
    print("RECEIVED VECTORS:")
    print(receivedVectors)

    # Decode
    decodedTuple = coder.decode(receivedVectors)
    decodedString = decodedTuple[1]
    print("DECODED STRING:")
    print(decodedString)

    if (decodedTuple[0]):
        io.writeFile(Lib.FILENAME_WRITE, decodedString)

        print("Same string? - " + repr(stringRead.encode(Lib.UTF_8)
                                       == decodedString.encode(Lib.UTF_8)))
    else:
        print(decodedTuple[1])
