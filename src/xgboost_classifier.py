#! /usr/bin/python

import xgboost as xgb
import os
import numpy as np
from utilities import load_data, save_results, aucScore
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn.grid_search import GridSearchCV

SEED  = 42

def fit(X, y):
	num_boost_round = num_boost_round
	label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
	dtrain = xgb.DMatrix(X, label=[label2num[label] for label in y])
	model = xgb.train(dtrain=dtrain)
	return model

def main():

	cwd = os.getcwd()
	trainDataLoc = cwd + '/../data/train.csv'
	testDataLoc = cwd + '/../data/test.csv'

	y, X = load_data(trainDataLoc)
	y_test, X_test = load_data(testDataLoc, use_labels=False)

	clf = xgb.XGBClassifier(max_depth=15, 
	                        n_estimators=200, learning_rate=.4, colsample_bytree=.8, seed=SEED)

	# fitting
	clf.fit(X, y, early_stopping_rounds=100, eval_metric="logloss", eval_set=[(X_test, y_test)])

	#print y_pred
	preds = clf.predict_proba(X_test)[:,1]

	save_results(preds, "XGBoost_classifier.csv")



if __name__ == '__main__':
	main()
