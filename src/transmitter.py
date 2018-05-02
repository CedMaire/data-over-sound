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

    decodedString = coder.decode(encodedVectors)
    print("DECODED STRING:")
    print(decodedString)

    io.writeFile(Lib.FILENAME_WRITE, decodedString)
