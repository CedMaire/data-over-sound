# Encode and decode byte/bit vectors. Chunk them in the right size, apply error recovery algorithms like
# Reed Salomon or Polar Code and "randomize" the byte/bit vectors to have P(0) = P(1) = 1/2 (Mult by huge
# prime + mod 256). + All the reverse operations.

# Encode: Apply ECC, Randomize, To bit vectors, Chunck
# Decode: Assemble, RecoverBits, to string, Recover ECC

import unireedsolomon as ReedSalomon
import lib as Lib


class Coder:
    # Constuctor, initializes the parameters.
    def __init__(self):
        self.RSCoder = ReedSalomon.rs.RSCoder(
            Lib.CODE_WORD_LENGTH, Lib.MESSAGE_LENGTH)

    def encode(self, string):
        # TODO:
        return string

    def decode(self, tupleList):
        # TODO:
        return tupleList

    # Chunks the 8-bit vectors into smaller vectors.
    def chunk(self, vectorList):
        # Chunk
        chuncked = list(map(lambda vector: [vector[i:i + Lib.CHUNK_SIZE] for i in range(0, len(vector), Lib.CHUNK_SIZE)],
                            vectorList))

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

    def randomizeBits(self, byteList):
        # TODO:
        return byteList

    def recoverBits(self, byteList):
        # TODO:
        return byteList

    # Converts a regular string into a list of 8-bit vectors.
    # "He" -> [[0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1]]
    def stringToListOfByteVectors(self, string):
        tmp = list(map(lambda x: bin(x)[2:], string.encode(Lib.UTF_8)))

        output = list()
        for e in tmp:
            test = list()
            for i in e:
                test.append(int(i))
            output.append(test)

        for e in output:
            if len(e) < 8:
                for i in range(8-len(e)):
                    e.insert(0, 0)

        return output

    # Converts a list of 8-bit vectors to a regular string (with accents).
    # [[0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1]] -> "He"
    def listOfByteVectorsToString(self, byteVectors):
        return bytes(list(
            map(lambda bitString: int(bitString, 2),
                map(lambda vector: "".join(
                    map(lambda x: repr(x), vector)), byteVectors)))).decode(
                        Lib.UNICODE_ESCAPE).encode(Lib.LATIN_1).decode(Lib.UTF_8)
