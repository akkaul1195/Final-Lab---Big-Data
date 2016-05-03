#! /usr/bin/python

import time
import sys, os
import numpy as np
from utilities import load_data, save_results
def main():
	filesToEnsemble = ['SVM_classifier.csv', 'LogisticRegressionResults.csv', 'sampleSubmission.csv']

	cwd = os.getcwd()

	for item in filesToEnsemble:
		item = cwd + '/../output/' + item
		f = open(item, 'w')
		print f
		print 'This is the first line of the ' + item + ': ' + f.readline()

if __name__ == '__main__':
    main()
