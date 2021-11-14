# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:25:07 2021

@author: jveverka
"""

import time
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

versions = [10, 25, 50, 75, 100, 125, 150, 175, 200]

for version in versions:

    train_trans_bovw = pd.read_csv('../data/train_orb_' + str(version) + '.csv')
    test_trans_bovw = pd.read_csv('../data/test_orb_' + str(version) + '.csv')
    
    y_train = train_trans_bovw.iloc[:,0]
    y_test = test_trans_bovw.iloc[:,0]
    X_train_trans_bovw = train_trans_bovw.iloc[:,1:]
    X_test_trans_bovw = test_trans_bovw.iloc[:,1:]
    
    print('[INFO] y_train dim:', y_train.shape)
    print('[INFO] y_test dim:', y_test.shape)
    print('[INFO] X_train_trans_bovw:', X_train_trans_bovw.shape)
    print('[INFO] X_test_trans_bovw dim:', X_test_trans_bovw.shape)
    
    ### model with scaler and hyperparameter tuning ###
    print('[INFO] model LR with scaler and hyperparameter tuning ', 'ORB ', str(version))
    start_time = time.time()
    
    steps = [('scaler', StandardScaler()),
              ('classifier', LogisticRegression(max_iter=10000))]
    
    params_space = {
        'classifier__solver': ['lbfgs', 'liblinear'],
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10],
    }
    
    pipeline = Pipeline(steps)
    
    gs_logit = GridSearchCV(estimator=pipeline,
                            param_grid=params_space,
                            scoring='accuracy',
                            cv=10,
                            verbose=0)
    
    gs_logit.fit(X_train_trans_bovw, y_train)
    
    for i in range(len(gs_logit.cv_results_['params'])):
        print(gs_logit.cv_results_['params'][i], 'cv Acc.:', gs_logit.cv_results_['mean_test_score'][i])
        
    print('[INFO] Best params: ', gs_logit.best_params_)
        
    
    y_pred = gs_logit.best_estimator_.predict(X_test_trans_bovw)
    
    print('[STATUS] Acc. score on test data: ', 
          gs_logit.best_estimator_.score(X_test_trans_bovw, y_test))
    conf_matrix = confusion_matrix(y_pred, y_test)
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    
    print(round(time.time() - start_time, 2))
