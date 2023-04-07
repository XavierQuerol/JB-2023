# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:21:47 2023

@author: xavid
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import numpy as np

def best_th_function(X,y,params):
    best_th = 0
    for i in range(10):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=i)
        model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:,1]
        # Array for finding the optimal threshold
        thresholds = np.arange(0.0, 1.0, 0.0001)
        fscore = np.zeros(shape=(len(thresholds)))
    
        # Fit the model
        for index, elem in enumerate(thresholds):
            # Corrected probabilities
            y_pred_prob = (y_pred > elem).astype('int')
            # Calculate the f-score
            fscore[index] = f1_score(y_test, y_pred_prob, average="macro")
    
        # Find the optimal threshold
        index = np.argmax(fscore)
        thresholdOpt = round(thresholds[index], ndigits = 4)
        fscoreOpt = round(fscore[index], ndigits = 4)
        print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))
        best_th += thresholdOpt
    best_th /= 10 
    return best_th