#! /usr/bin/python

import time
import sys, os
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from utilities import load_data, save_results
from sklearn import (metrics, cross_validation, preprocessing)

SEED = 23

def main():

    cwd = os.getcwd()
    trainDataLoc = cwd + '/../data/train.csv'
    testDataLoc = cwd + '/../data/test.csv'

    y, X = load_data(trainDataLoc)
    y_test, X_test = load_data(testDataLoc, use_labels=False)

    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)
    
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=.20, random_state=SEED)


    model = svm.SVC(C=1, probability=True, kernel='linear')
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:, 1]

    # compute AUC metric for this CV fold
    fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print "AUC : %f" % (roc_auc)

    preds = model.predict_proba(X_test)[:, 1]

    save_results(preds, "SVM_classifier.csv")




if __name__ == '__main__':
    main()
