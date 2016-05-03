#! /usr/bin/python

import time
import sys, os
import numpy as np
import collections
from utilities import load_data, save_results

numItems = 58921

def main():
	filesToEnsemble = ['SVM_classifier.csv', 'LogisticRegressionResults.csv']#, 'sampleSubmission.csv']
	fdata = []
	itemSum = [numItems]
	cwd = os.getcwd()

	for item in filesToEnsemble:
		item = cwd + '/../output/' + item
		fdict = {}
		f = open(item, 'r')
		first = True
		for line in f:
			if not first:
				contents = line.split(',', 2)
				contents[1] = contents[1].strip('\n')
				fdict[int(contents[0])] = float(contents[1])
			else:
				first = False
				continue
		fOrdered = collections.OrderedDict(sorted(fdict.items(), key=lambda t: t[1]))
		fdata.append(fOrdered)

	for d in fdata:
		print d.keys()[:5]
		for j in range(58921):
			itemSum[j] = (itemSum[j], d[j+1])


if __name__ == '__main__':
    main()
