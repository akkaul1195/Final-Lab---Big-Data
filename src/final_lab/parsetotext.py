#!/usr/bin/env python
import sys

f = open('train.csv','r')
lines = f.readlines()
file = open('train.txt', 'w')
for i, line in enumerate(lines):
	if (i!=0):
		words = line.split(',')
		for j, word in enumerate(words):
			if (j != 9):
				file.write(word.strip(',') + " ")
			else:
				file.write(word.strip(','))
file.close()
f.close()
print ('finished')