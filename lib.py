# To be deleted or used for external parameters used everywhere.

import unireedsolomon as ReedSalomon

UTF_8 = "utf-8"
CODE_WORD_LENGTH = 255
MESSAGE_LENGTH = 180

RSCoder = ReedSalomon.rs.RSCoder(CODE_WORD_LENGTH, MESSAGE_LENGTH)


def RSEncode(stringToEncode):
    return RSCoder.encode(stringToEncode)


def RSDecode(stringToDecode):
    return RSCoder.decode(stringToDecode)
