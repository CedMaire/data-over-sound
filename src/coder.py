# Encode and decode byte/bit vectors. Chunk them in the right size, apply error recovery algorithms like
# Reed Salomon or Polar Code and "randomize" the byte/bit vectors to have P(0) = P(1) = 1/2 (Mult by huge
# prime + mod 256). + All the reverse operations.

# Encode: Apply ECC, Randomize, Chunck
# Decode: Assemble, RecoverBits, Recover ECC

import unireedsolomon as ReedSalomon

UTF_8 = "utf-8"
CODE_WORD_LENGTH = 255
MESSAGE_LENGTH = 180

RSCoder = ReedSalomon.rs.RSCoder(CODE_WORD_LENGTH, MESSAGE_LENGTH)


class Coder:
    # Constuctor, initializes the parameters.
    def __init__(self):
        pass

    def encode(self, string):
        return string

    def decode(self, tupleList):
        return tupleList

    def chunk(self, tupleList):
        return tupleList

    def assemble(self, tupleList):
        return tupleList

    def applyEcc(self, string):
        return RSCoder.encode(string)

    def recoverEcc(self, string):
        return RSCoder.decode(string)

    def randomizeBits(self, tupleList):
        return tupleList

    def recoverBits(self, tupleList):
        return tupleList
