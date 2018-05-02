import lib as Lib
import numpy as np

np.random.seed(42)
print(np.random.randint(20, size=(20)))


class IODeux:
    def __init__(self):
        pass

    def readFile(self, fileName):
        file = open(file=fileName, mode="r", encoding=Lib.UTF_8)
        string = file.read()
        file.close()

        print("Read:\n" + string)

        return string

    def writeFile(self, fileName, string):
        file = open(file=fileName, mode="w", encoding=Lib.UTF_8)
        file.write(string)
        file.close()

        print("Write:\n" + string)

        return True
