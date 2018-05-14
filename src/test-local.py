import iodeux as IODeux
import lib as Lib
import coder as Coder

if __name__ == "__main__":
    io = IODeux.IODeux()
    coder = Coder.Coder()

    stringRead = io.readFile(Lib.FILENAME_READ)

    encodedVectors = coder.encode(stringRead)
    print("ENCODED VECTORS:")
    print(encodedVectors)

    decodedTuple = coder.decode(encodedVectors)
    decodedString = decodedTuple[1]
    print("DECODED STRING:")
    print(decodedString)

    if (decodedTuple[0]):
        io.writeFile(Lib.FILENAME_WRITE, decodedString)

        print("Same string? - " + repr(stringRead == decodedString))
    else:
        print(decodedTuple[1])

    # ###################################################################
    # Test Local
    # ###################################################################
    '''
    synthesizer = Synthesizer.Synthesizer()

    noise1 = synthesizer.createWhiteNoise()
    noise2 = synthesizer.createWhiteNoise()

    a = [[0], [1], [1], [0], [0], [1], [0], [0], [0], [1], [1], [0], [0], [1], [0], [0],
         [0], [1], [1], [0], [0], [1], [0], [0], [0], [1], [1], [0], [0], [1], [0], [0]]
    length = len(a)

    nonoise = 2
    if(nonoise == 2):
        Noise3 = Noise.band_limited_Noise(
            2000, 3000, Lib.FS*Lib.TIME_BY_CHUNK * length, Lib.FS)*100000
    else:
        Noise3 = Noise.band_limited_Noise(
            1000, 2000, Lib.FS*Lib.TIME_BY_CHUNK * length, Lib.FS)*100000
    signal = generateCompleteSignal(a, nonoise)
    signal = signal+Noise3
    signal += Noise3
    midSignal = Numpy.concatenate([noise1, signal])
    fullSignal = Numpy.concatenate([noise2, midSignal])
    sync = sync(fullSignal, length)
    decodeSignal(signal, nonoise)
    '''
