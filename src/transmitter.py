# Main transmitter program.

import lib as Lib

if __name__ == '__main__':
    file = open(file="text.txt", mode="r", encoding=Lib.UTF_8)

    print()

    string = file.read()
    file.close()

    print(repr(string) + "\n")
    # printer += "0" * offset // Use this to add 0's at the beginning to always have exactly 8 bits.
    # print(repr(list(map(lambda x: bin(x)[2:], list(
    #    string.encode(Lib.UTF_8))))) + "\n")

    stringEncoded = Lib.RSEncode(string)
    print(repr(stringEncoded) + "\n")

    stringDecoded = Lib.RSDecode(stringEncoded)[0]
    # stringDecoded = Lib.RSDecode(
    #    "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Hello World! zend me to te comuter in frint of me. hdjdkffkdmndndjhvuhzwazgaiuhhfdauhfijossjgfshbghsjuzagdzgwadgfseiufhsehfsieofohsuhfsuhfeiuhfaefesfsefsfefs\ntl\x99\x9d¥·bXx\x01e\x92\x16¡\x01^vÚæ0\x12\rá\x9bGÔ)\x8e\x88\x95\x8fê\x98Û#\x895DâÑcËkZ\x19\x1d2êgqB\x07\x1e\x88¬r*æe\x9cÑqÁ)\x8eV¹~U#cú5H/")[0]
    print(repr(stringDecoded) + "\n")

    print("Original equals decoded? - " + repr(string == stringDecoded) + "\n")
