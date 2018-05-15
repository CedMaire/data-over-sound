import iodeux as IODeux
import lib as Lib
import coder as Coder
import synth as Synthesizer
import noisedeux as Noise
import numpy as Numpy

if __name__ == "__main__":
    io = IODeux.IODeux()
    coder = Coder.Coder()
    synthesizer = Synthesizer.Synthesizer()
    noise = Noise.NoiseGenerator()

    stringRead = io.readFile(Lib.FILENAME_READ)
    print(len(stringRead))

    # Encode
    encodedVectors = coder.encode(stringRead)
    print(len(encodedVectors))
    print("ENCODED VECTORS:")
    print(encodedVectors)

    # Generate Signal
    noiseStart = synthesizer.createWhiteNoise()
    noiseEnd = synthesizer.createWhiteNoise()

    noNoise = 1  # CHANGE AS YOU WANT BETWEEN {1, 2}
    noiseMiddle = None
    if(noNoise == 2):
        print(Lib.NEEDED_AMOUNT_OF_VECTORS)
        print(Lib.SAMPLES_PER_SEC * Lib.TIME_PER_CHUNK *
              Lib.NEEDED_AMOUNT_OF_VECTORS)
        noiseMiddle = noise.generateBandLimitedNoise(
            Lib.UPPER_LOW_FREQUENCY_BOUND,
            Lib.UPPER_UPPER_FREQUENCY_BOUND,
            Lib.SAMPLES_PER_SEC * Lib.TIME_PER_CHUNK * Lib.NEEDED_AMOUNT_OF_VECTORS,
            Lib.SAMPLES_PER_SEC) * 100000
    else:
        print(Lib.NEEDED_AMOUNT_OF_VECTORS)
        print(Lib.SAMPLES_PER_SEC * Lib.TIME_PER_CHUNK *
              Lib.NEEDED_AMOUNT_OF_VECTORS)
        noiseMiddle = noise.generateBandLimitedNoise(
            Lib.LOWER_LOW_FREQUENCY_BOUND,
            Lib.LOWER_UPPER_FREQUENCY_BOUND,
            Lib.SAMPLES_PER_SEC * Lib.TIME_PER_CHUNK * Lib.NEEDED_AMOUNT_OF_VECTORS,
            Lib.SAMPLES_PER_SEC) * 100000

    signalToSend = synthesizer.generateCompleteSignal(encodedVectors, noNoise)

    # Send
    print(signalToSend.shape)
    print(noiseMiddle.shape)
    signalToSend += noiseMiddle
    signalToSend = Numpy.concatenate(
        [noiseStart, signalToSend, noiseEnd])

    # Receive
    dataSignal = synthesizer.extractDataSignal(signalToSend)
    encodedVectors = synthesizer.decodeSignalToBitVectors(dataSignal, noNoise)

    # Decode
    decodedTuple = coder.decode(encodedVectors)
    decodedString = decodedTuple[1]
    print("DECODED STRING:")
    print(decodedString)

    if (decodedTuple[0]):
        io.writeFile(Lib.FILENAME_WRITE, decodedString)

        print("Same string? - " + repr(stringRead == decodedString))
    else:
        print(decodedTuple[1])
