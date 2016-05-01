#!/usr/bin/env python
import sys
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score


def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

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

f.close()
t = open ('test.txt', 'r')
lines = t.readlines()
test = []
for i, line in enumerate(lines):
    line = line.strip()
    # split the line into words
    words = line.split()
    Array = []
    for j, word in enumerate(words):
    	if (j != 0):
        	Array.append(word)
    test.append(Array)
print ('finished part 1')
clf = AdaBoostClassifier(n_estimators=1000, learning_rate=2)
scores = clf.fit( Matrix, salary)
prediction = scores.predict_proba(test)[:, 1]
create_test_submission('output.csv', prediction)
print ("done")


score = cross_val_score(clf, Matrix, salary)
print score
print score.mean()  
