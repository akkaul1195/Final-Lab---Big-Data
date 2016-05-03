#! /usr/bin/python

import time
import sys, os
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from utilities import load_data, save_results
from sklearn import (metrics, cross_validation, preprocessing)
from sklearn.grid_search import GridSearchCV

SEED = 23

def aucScore(model, x, y):
	preds = model.predict_proba(x)[:, 1]
	fpr, tpr, thresholds = metrics.roc_curve(y, preds)
	roc_auc = metrics.auc(fpr, tpr)
	return roc_auc

def findBestModel(x,y):
	

#These were the best params found: {'kernel': 'rbf', 'C': 1, 'random_state': 23, 'probability': True, 'gamma': 1}
#They received an auc score of: 0.860507309069


	#Testing each param change independently instead
	#params = [{'kernel': ['linear'], 'C': [.01, 1, 10, 100, 1000], 'probability': [True], 'random_state': [SEED]},
	#				{'kernel': ['rbf'], 'C': [.01, 1, 10, 100, 1000], 'gamma': [.0001, .01, 1], 'probability': [True], 'random_state': [SEED]}]

	params = [{'kernel': ['rbf'], 'C': [1], 'gamma': [1, 10, 100], 'probability': [True], 'random_state': [SEED]}]


	#3 fold cross validation
	svmGS = GridSearchCV(svm.SVC(), params, n_jobs=8, cv=3, scoring=aucScore)
	svmGS.fit(x, y)
	print 'These were the best params found: ' + str(svmGS.best_params_)
	print 'They received an auc score of: ' + str(svmGS.best_score_)

	return svmGS.best_estimator_


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
    

    model = findBestModel(X, y)
    """
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=.20, random_state=SEED)
    
    model = svm.SVC(C=1, probability=True, kernel='rbf')
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:, 1]

    # compute AUC metric for this CV fold
    fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print "AUC : %f" % (roc_auc)
    """

    preds = model.predict_proba(X_test)[:, 1]

    save_results(preds, "SVM_classifier.csv")


if __name__ == '__main__':
    main()


