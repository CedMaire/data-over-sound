# Main transmitter program.

import iodeux as IO
import lib as Lib

if __name__ == '__main__':
    io = IO.IO()

    stringRead = io.readFile(Lib.FILENAME_READ)

    io.writeFile(Lib.FILENAME_WRITE, stringRead)
