# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:53:30 2021

@author: jveverka
"""

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('../train_data_full.csv')
test_data = pd.read_csv('../test_data_full.csv')

y_train = train_data.iloc[:,0]
y_test = test_data.iloc[:,0]
X_train = train_data.iloc[:,1:]
X_test = test_data.iloc[:,1:]

print('[INFO] y_train dim:', y_train.shape)
print('[INFO] y_test dim:', y_test.shape)
print('[INFO] X_train:', X_train.shape)
print('[INFO] X_train dim:', X_test.shape)

### basic model ###
start_time = time.time()
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(lr.score(X_test, y_test))

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(round(time.time() - start_time, 2))

### model with scaler ###
start_time = time.time()
steps = [('scaler', StandardScaler()),
          ('classifier', LogisticRegression(max_iter=10000))]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(pipeline.score(X_test, y_test))

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(round(time.time() - start_time, 2))

### model with scaler and hyperparameter tuning ###
start_time = time.time()
steps = [('scaler', StandardScaler()),
          ('classifier', LogisticRegression(max_iter=10000))]

params_space = {
    'classifier__solver': ['lbfgs', 'liblinear'],
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10],
    'classifier__penalty': ['l2']
}

pipeline = Pipeline(steps)

gs_logit = GridSearchCV(estimator=pipeline,
                        param_grid=params_space,
                        scoring='neg_root_mean_squared_error',
                        cv=10,
                        verbose=0)

gs_logit.fit(X_train, y_train)

for i in range(len(gs_logit.cv_results_['params'])):
    print(gs_logit.cv_results_['params'][i], 'test RMSE:', gs_logit.cv_results_['mean_test_score'][i])

y_pred = gs_logit.best_estimator_.predict(X_test)

print(gs_logit.best_estimator_.score(X_test, y_test))

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(gs_logit.best_estimator_.score(X_test, y_test))
print(round(time.time() - start_time, 2))

### model with scaler and hyperparameter tuning + PCA ###
from sklearn.decomposition import PCA

start_time = time.time()
steps = [('scaler', StandardScaler()),
          ('pca', PCA()),
          ('classifier', LogisticRegression(max_iter=10000))]

params_space = {
    'pca__n_components': [5, 10, 15, 20, 30, 40, 50, 60, 120],
    'classifier__solver': ['lbfgs', 'liblinear'],
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10],
    'classifier__penalty': ['l2']
}

pipeline = Pipeline(steps)

gs_logit_pca = GridSearchCV(estimator=pipeline,
                        param_grid=params_space,
                        scoring='neg_root_mean_squared_error',
                        cv=10,
                        verbose=0)

gs_logit_pca.fit(X_train, y_train)

for i in range(len(gs_logit_pca.cv_results_['params'])):
    print(gs_logit_pca.cv_results_['params'][i], 'test RMSE:', gs_logit_pca.cv_results_['mean_test_score'][i])

y_pred = gs_logit_pca.best_estimator_.predict(X_test)

print(gs_logit_pca.best_estimator_.score(X_test, y_test))

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred))
print(round(time.time() - start_time, 2))