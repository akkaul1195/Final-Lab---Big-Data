#! /usr/bin/python

import time
import sys, os
import numpy as np
import collections
from utilities import load_data, save_results

numItems = 58921

def main():
	filesToEnsemble = ['SVM_classifier.csv', 'output.csv', 'XGBoost_classifier.csv']#, 'LogisticRegressionResults.csv','logistic_regression_pred.csv']#, 'sampleSubmission.csv']
	fdata = []
	itemSum = []
	for i in range(numItems):
		itemSum.append((i,0))
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

		#fList is a list of (row, probability) tuples
		fList = sorted(fdict.items(), key=lambda t: t[1]) 
		for i in range(len(fList)):
			fList[i] = (fList[i][0], i) # fList is now a list of (row, rank)

		fdata.append(fList)


	for l in fdata:
		for item in l:
			row = item[0]
			itemSum[row-1] = (itemSum[row-1][0], itemSum[row-1][1] + item[1])

	reSorted = sorted(itemSum, key=lambda t: t[1])

	for i in range(len(reSorted)):
		reSorted[i] = (reSorted[i][0] + 1, float(i) / numItems)

	backToNormal = sorted(reSorted, key=lambda t: t[0])

	save_results(map(lambda x: x[1], backToNormal), 'Ensembled_Results.csv')

if __name__ == '__main__':
    main()
