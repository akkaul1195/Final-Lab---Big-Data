#!/usr/bin/env python
import sys

f = open('train.csv','r')
file = open('train.txt', 'w')
for i, line in enumerate(f.readlines()):
	if (i!=0):
		words = line.split(',')
		for j, word in enumerate(words):
			file.write(word.strip(',') + " ")
		file.write("\n")
file.close()
f.close()
print ('finished')