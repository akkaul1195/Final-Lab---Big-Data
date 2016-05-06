#! /usr/bin/python

""" Amazon Access Challenge Starter Code

These files provide some starter code using 
the scikit-learn library. It provides some examples on how
to design a simple algorithm, including pre-processing,
training a logistic regression classifier on the data,
assess its performance through cross-validation and some 
pointers on where to go next.

Paul Duan <email@paulduan.com>
"""

from __future__ import division
import os
import numpy as np
from utilities import load_data, save_results, aucScore
from sklearn import (metrics, cross_validation, linear_model, preprocessing)

SEED = 42  # always use a seed for randomized procedures

def findBestModel(x,y):
    
#These were the best params found:
#They received an auc score of:


    #Testing each param 
    params = []

    #3 fold cross validation
    svmGS = GridSearchCV(linear_model.LogisticRegression(), params, n_jobs=-1, cv=3, scoring=aucScore)
    svmGS.fit(x, y)
    print 'These were the best params found: ' + str(svmGS.best_params_)
    print 'They received an auc score of: ' + str(svmGS.best_score_)

    return svmGS.best_estimator_


def main():
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    model = linear_model.LogisticRegression(C=3)  # the classifier we'll use

    # === load data in memory === #
    print "loading data"

    cwd = os.getcwd()
    trainDataLoc = cwd + '/../data/train.csv'
    testDataLoc = cwd + '/../data/test.csv'

    y, X = load_data(trainDataLoc)
    y_test, X_test = load_data(testDataLoc, use_labels=False)

    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)

    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

    # === training & metrics === #
    mean_auc = 0.0
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=i*SEED)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        # train model and make predictions
        model.fit(X_train, y_train) 
        preds = model.predict_proba(X_cv)[:, 1]

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc

    print "Mean AUC: %f" % (mean_auc/n)

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]
    #filename = raw_input("Enter name for submission file: ")
    filename = 'LogisticRegressionResults'
    save_results(preds, filename + ".csv")

if __name__ == '__main__':
    main()
