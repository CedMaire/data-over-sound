import lib as Lib
import numpy as Numpy
import os as OS


class IODeux:
    def __init__(self):
        pass

    def readFile(self, fileName):
        dirname, _ = OS.path.split(OS.path.abspath(__file__))
        dirname += "/../"

        file = open(file=dirname + fileName, mode="r", encoding=Lib.UTF_8)
        string = file.read()
        file.close()

        print("READ:")
        print(string)

        return string

    def writeFile(self, fileName, string):
        dirname, _ = OS.path.split(OS.path.abspath(__file__))
        dirname += "/../"

        file = open(file=dirname + fileName, mode="w", encoding=Lib.UTF_8)
        file.write(string)
        file.close()

        print("WRITTEN:")
        print(string)

        return True
