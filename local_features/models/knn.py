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
from sklearn.neighbors import KNeighborsClassifier

versions = [10, 25, 50, 75, 100, 125, 150, 175, 200]

for version in versions:

    train_trans_bovw = pd.read_csv('../data/train_sift_' + str(version) + '.csv')
    test_trans_bovw = pd.read_csv('../data/test_sift_' + str(version) + '.csv')

    y_train = train_trans_bovw.iloc[:,0]
    y_test = test_trans_bovw.iloc[:,0]
    X_train_trans_bovw = train_trans_bovw.iloc[:,1:]
    X_test_trans_bovw = test_trans_bovw.iloc[:,1:]
    
    print('[INFO] y_train dim:', y_train.shape)
    print('[INFO] y_test dim:', y_test.shape)
    print('[INFO] X_train_trans_bovw:', X_train_trans_bovw.shape)
    print('[INFO] X_test_trans_bovw dim:', X_test_trans_bovw.shape)
    
    ### model with scaler and hyperparameter tuning ###
    print('[INFO] model KNN with scaler and hyperparameter tuning ', 'SIFT ', str(version))
    start_time = time.time()
    
    steps = [('scaler', StandardScaler()),
             ('classifier', KNeighborsClassifier())]
    
    params_space = {
        'classifier__n_neighbors': [1, 3, 5, 7, 10, 15, 20, 25, 30],
        'classifier__leaf_size': [1, 2, 3, 4, 5, 10],
    }
    
    pipeline = Pipeline(steps)
    
    gs_knn = GridSearchCV(estimator=pipeline,
                            param_grid=params_space,
                            scoring='accuracy',
                            cv=10,
                            verbose=0)
    
    gs_knn.fit(X_train_trans_bovw, y_train)
    
    for i in range(len(gs_knn.cv_results_['params'])):
        print(gs_knn.cv_results_['params'][i], 'cv Acc.:', gs_knn.cv_results_['mean_test_score'][i])
        
    print('[INFO] Best params: ', gs_knn.best_params_)
        
    
    y_pred = gs_knn.best_estimator_.predict(X_test_trans_bovw)
    
    print('[STATUS] Acc. score on test data: ', 
    gs_knn.best_estimator_.score(X_test_trans_bovw, y_test))
    conf_matrix = confusion_matrix(y_pred, y_test)
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    
    print(round(time.time() - start_time, 2))
