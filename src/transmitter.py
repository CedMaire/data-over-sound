# Main transmitter program.

import iodeux as IODeux
import lib as Lib
import coder as Coder

if __name__ == '__main__':
    '''
    io = IODeux.IODeux()
    stringRead = io.readFile(Lib.FILENAME_READ)
    io.writeFile(Lib.FILENAME_WRITE, stringRead)
    '''
    stringToSend = "Héllò World!\n"
    coder = Coder.Coder()
    rsEncodedString = coder.applyEcc(stringToSend)
    print(repr(rsEncodedString))
    byteVectorList = coder.stringToListOfByteVectors(rsEncodedString)
    print(byteVectorList)
    chunkedVectorsList = coder.chunk(byteVectorList)
    print(chunkedVectorsList)

    # SEND

    assembledVectorsList = coder.assemble(chunkedVectorsList)
    print(assembledVectorsList)
    stringReceived = coder.listOfByteVectorsToString(assembledVectorsList)
    print(repr(stringReceived))
    rsDecodedString = coder.recoverEcc(stringReceived)
    print(rsDecodedString)
