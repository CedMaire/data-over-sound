import unireedsolomon as ReedSalomon
import lib as Lib
# import mpmath as BigNumbers
import numpy as Numpy


class Coder:
    # Constuctor, initializes the parameters.
    def __init__(self):
        self.RSCoder = ReedSalomon.rs.RSCoder(
            Lib.CODE_WORD_LENGTH, Lib.MESSAGE_LENGTH)
        self.receivedData = list()

    # TODO: Test
    def newVectorReceived(self, vector):
        self.receivedData.append(vector)
        length = len(self.receivedData)

        while (self.dataReceivedIsTooLong(length)):
            self.receivedData.pop(0)

        if (self.dataReceivedHasRightLength(length)):
            return self.decode(self.receivedData)
        else:
            print(Lib.DECODING_NOT_READY)
            return (False, Lib.DECODING_NOT_READY)

    # Checks if the received data is too long.
    def dataReceivedIsTooLong(self, length):
        return length > Lib.NEEDED_AMOUNT_OF_CHUNKS

    # Checks if the received data has the correct lenght.
    def dataReceivedHasRightLength(self, length):
        return length == Lib.NEEDED_AMOUNT_OF_CHUNKS

    # Encodes the string so that it can be sent as k bits at a time.
    def encode(self, string):
        rsEncodedString = self.applyEcc(string)
        byteVectorList = self.stringToListOfByteVectors(rsEncodedString)
        chunkedVectorsList = self.chunk(byteVectorList)

        return chunkedVectorsList

    # Decodes k-bit vectors to get back a readable string.
    def decode(self, tupleList):
        output = Lib.DECODING_FAILED
        isDecoded = False

        try:
            assembledVectorsList = self.assemble(tupleList)
            stringReceived = self.listOfByteVectorsToString(
                assembledVectorsList)
            output = self.recoverEcc(stringReceived)
            isDecoded = True
        except:
            print(Lib.DECODING_FAILED)

        return (isDecoded, output)

    # Chunks the 8-bit vectors into smaller vectors.
    def chunk(self, vectorList):
        # Chunk
        chuncked = list(map(lambda vector: [vector[i:i + Lib.CHUNK_SIZE]
                                            for i in range(0, len(vector), Lib.CHUNK_SIZE)], vectorList))

        # Flatten
        outputList = list()
        for i in range(0, len(chuncked)):
            for j in range(0, len(chuncked[i])):
                outputList.append(chuncked[i][j])

        return outputList

    # Assembles the k-bit vectors to have 8-bit vectors again.
    def assemble(self, vectorList):
        output = list()
        step = int(Lib.BYTE_BIT_SIZE / Lib.CHUNK_SIZE)

        for i in range(0, len(vectorList), step):
            tempList = list()
            for j in range(0, step):
                tempList.append(vectorList[i + j])
            output.append([bit for vector in tempList for bit in vector])

        return output

    # Applies an error correcting encoding. In this case it is the Reed Solomon ECC.
    def applyEcc(self, string):
        return self.RSCoder.encode(string)

    # Tries to recover the original string from a received Reed Solomon ECC.
    def recoverEcc(self, string):
        return self.RSCoder.decode(string)[0]

    # Randomizes the bytes to expect having a P(0)=P(1)=1/2 so that we can use an ML rule.
    def randomizeBytes(self, byteString):
        '''
        output = map(lambda x: int.from_bytes(
            [x], byteorder=Lib.BYTE_ENDIANESS) + 1, byteString)
        output = map(lambda x: BigNumbers.fmul(
            x, Lib.BIG_PRIME_NUMBER), output)
        output = map(lambda x: BigNumbers.fmod(
            x, Lib.BYTE_DIFF_VALUES), output)
        output = map(lambda x: int(BigNumbers.nstr(
            x)[: - 2]), output)
        '''
        output = map(lambda x: Lib.BYTE_RANDOMIZE_MAP.get(int.from_bytes(
            [x], byteorder=Lib.BYTE_ENDIANESS)), byteString)

        return bytes(output)

    # Recovers the original bytes that have been randomized.
    def recoverBytes(self, byteString):
        output = map(lambda x: Lib.BYTE_RECOVER_MAP.get(int.from_bytes(
            [x], byteorder=Lib.BYTE_ENDIANESS)), byteString)

        return bytes(output)

    # Converts a regular string into a list of 8-bit vectors.
    # "He" -> [[0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1]]
    def stringToListOfByteVectors(self, string):
        tmp = list(
            map(lambda x: bin(x)[2:], self.randomizeBytes(string.encode(Lib.LATIN_1))))

        output = list()
        for e in tmp:
            test = list()
            for i in e:
                test.append(int(i))
            output.append(test)

        for e in output:
            if len(e) < 8:
                for i in range(8 - len(e)):
                    e.insert(0, 0)

        return output

    # Converts a list of 8-bit vectors to a regular string (with accents).
    # [[0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1]] -> "He"
    def listOfByteVectorsToString(self, byteVectors):
        return self.recoverBytes(bytes(list(
            map(lambda bitString: int(bitString, 2),
                map(lambda vector: "".join(
                    map(lambda x: repr(x), vector)), byteVectors))))).decode(
                        Lib.UNICODE_ESCAPE).encode(Lib.LATIN_1).decode(Lib.LATIN_1)

    # Creates a random array using a seed.
    def createRandomArray(self, array_size, seed, max_value):
        Numpy.random.seed(seed)
        tmp = Numpy.random.randint(max_value, size=(array_size))
        return tmp
