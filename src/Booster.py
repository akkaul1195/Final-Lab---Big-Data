#!/usr/bin/env python
import sys, os
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn import (metrics, cross_validation, preprocessing)
from utilities import load_data, save_results


####    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
####    This is the URL for the parameters for AdaBoost Classifier

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'


cwd = os.getcwd()
trainDataLoc = cwd + '/../data/train.csv'
testDataLoc = cwd + '/../data/test.csv'
y, X = load_data(trainDataLoc)
y_test, X_test = load_data(testDataLoc, use_labels=False)
print ("encoding")
encoder = preprocessing.OneHotEncoder()
print ("fitting")
encoder.fit(np.vstack((X, X_test)))
X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
X_test = encoder.transform(X_test)

print("about to classify")
clf = AdaBoostClassifier(base_estimator=None, n_estimators=900, learning_rate=1.8)
scores = clf.fit(X, y)
    # """
    # X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=.20, random_state=SEED)
    
    # model = svm.SVC(C=1, probability=True, kernel='rbf')
    # model.fit(X_train, y_train)
    # preds = model.predict_proba(X_cv)[:, 1]

    # # compute AUC metric for this CV fold
    # fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
    # roc_auc = metrics.auc(fpr, tpr)
    # print "AUC : %f" % (roc_auc)
    # """

prediction = scores.predict_proba(X_test)[:, 1]

create_test_submission('output.csv', prediction)
print ("done")



# score = cross_val_score(clf, Matrix, salary)
# print score
# print score.mean()  
