#!/usr/bin/env python
import sys
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm


f = open('train.txt', 'r')
lines = f.readlines()
Matrix = []
access = []
for i, line in enumerate(lines):
    line = line.strip()
    # split the line into words
    words = line.split()
    Array = []
    for j, word in enumerate(words):
        if (j == 0):
            access.append(word)
        else:
            Array.append(word)
    Matrix.append(Array)


t = open ('test.txt', 'r')
lines = t.readlines()
test = []
for i, line in enumerate(lines):
    line = line.strip()
    # split the line into words
    words = line.split()
    Array = []
    for word in words:
        Array.append(word)
    test.append(Array)
print ('finished part 1')

clf = svm.SVC(kernel='rbf', C=1).fit(Matrix, access)
svm.SVC.clf.Predict(test)



print (clf.score(X_test, y_test))