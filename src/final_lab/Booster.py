#!/usr/bin/env python
import sys
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score


f = open('train.txt', 'r')
lines = f.readlines()
Matrix = []
salary = []
for i, line in enumerate(lines):
    line = line.strip()
    # split the line into words
    words = line.split()
    Array = []
    for j, word in enumerate(words):
        if (j == 0):
            salary.append(word)
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
print (test)
print ('finished part 1')
clf = AdaBoostClassifier(n_estimators=400, learning_rate=1.75)
predict = cross_val_predict(clf, test)
a = open('answers.csv', 'w')
for i, prediction in enumerate(predict):
	a.write(i+', '+prediction+ '\n')
	print(i+', '+prediction+ '\n')
print ('done')


#scores = cross_val_score(clf, Matrix, salary)
#print scores
#print scores.mean()  
