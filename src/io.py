# Read and write from to a text file. Convert the text into byte vectors. Ex: a -> (1,1,1,1,0,0,1,0)
# Convert byte vectors back to text.

import lib as Lib

if __name__ == '__main__':
	file = open(file="../text.txt", mode="r", encoding=Lib.UTF_8)
	string = file.read()
	file.close()

	print(string)

	tmp = list(map(lambda x : bin(x)[2:], string.encode(Lib.UTF_8)))

	#print(string.encode(Lib.UTF_8))
	#print(tmp)
	#print(len(string))
	#print(len(tmp))

	output = list()
	for e in tmp:
		test = list()
		for i in e:
			test.append(int(i))
		output.append(test)
	
	for e in output:
		if len(e)<8:
			for i in range(8-len(e)):
				e.insert(0,0)

	#print(len(output))
	print(output)
	



