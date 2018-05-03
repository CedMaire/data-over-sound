import iodeux as IODeux
import lib as Lib
import coder as Coder

if __name__ == '__main__':
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
