#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:03:53 2017

@author: XiHUANG
"""

import math
import os
import random
import pandas as pd
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreeRegression
from sklearn.ensemble import RandomForestRegression

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import backend as K
from keras.callbacks import EarlyStopping
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def xgboost(x,v,y,v_y):
    dtrain = xgb.DMatrix(x, label=y)
    dtest = xgb.DMatrix(v, label=v_y)
    evallist  = [(dtrain,'train'), (dtest,'test')]

    nrounds = 10000
    e1 = 20
    e2 = 40
    lambda1 = np.ones(e1)*0.01
    lambda2 = np.ones(e2)*0.01
    lambda3 = np.ones(nrounds - e1 - e2)*0.01
    learning_rates = np.hstack([lambda1,lambda2,lambda3])
    
    param = {'max_depth':10,
             'eta':0.01,
             'min_child_weight':1,
             'max_delta_step':0,
             'gamma':1,
             'lambda':1,
             'alpha':3,
             'colsample_bytree':0.3,
             'subsample':1,
             'eval_metric':'rmse',
             'maximize':False,
             'nthread':4}
    
    xgb_fit = xgb.train(param, 
                        dtrain, 
                        nrounds, 
                        evals = evallist, 
                        early_stopping_rounds = 20, 
                        verbose_eval = 50, 
                        learning_rates = learning_rates.tolist())
    return xgb_fit
    
def modellist(listname,x,y):
    modelresultlist=[]
    for item in listname:
        if item =='randomforest':
            model = RandomForestRegressor(n_estimators = 100, oob_score = True,n_jobs = 1,random_state =1)
            treemodel=model.fit(x,y)
        if item =='extratree':
            model=ExtraTreesRegressor(n_estimators = 100,n_jobs = 1,random_state =1).fit(x, y)
        if item == 'NN':
            model = Sequential()

            model.add(Dense(64, input_dim=x.shape[1], init='uniform')) 
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
            
            model.add(Dense(1, init='uniform'))
            model.add(Activation('linear'))
            
            model.compile(loss=root_mean_squared_error, optimizer='rmsprop')
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')
            model=model.fit(x, y,
                      nb_epoch=50, batch_size=128, verbose=2,
                      validation_data=(X_train_test.values, y_train_test),
                      callbacks=[early_stopping])
        if item == 'lasso':
            clf = Lasso(alpha=100)
            Model = clf.fit(x, y) 
        if item == 'ridge':
            clf = Ridge(alpha=10)
            Model = clf.fit(x, y)
            
            
          
            
    modelresultlist.append(model)
    return modelresultlist
    
def prediction(model,x):
    preds=model.predict(x)
    preds[preds>3] = 3
    preds[preds<1] = 1
    return preds
    
def rms(y,pred):
    rms = sqrt(mean_squared_error(y_train_test, Lasso_pred))
return rms
      
with open('feats.pkl', 'rb') as infile:
    X = pickle.load(infile)

with open('target.pkl', 'rb') as infile:
    y = pickle.load(infile)

n_train = 74067 # Number of rows containing training data
X = X.fillna(0)
# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
X.iloc[:,2:] = min_max_scaler.fit_transform(X.iloc[:,2:])

X.iloc[:,2:] = preprocessing.scale(X.iloc[:,2:])
X_train = X.iloc[:n_train,:]
X_test = X.iloc[n_train:,:]

y_train = y.iloc[:n_train,1]

X_id = X[['id', 'product_uid']].iloc[:n_train,:]
trainend = int(len(X_id)*0.885)
counts = X_id[:trainend].groupby(['product_uid']).count()[['id']]

# Only care about uid's with counts higher than 1 (do not remove single rows)
counts = counts[counts['id'] > 1]
counts = counts.add_suffix('_Count').reset_index()
valid_product_uids = set(counts['product_uid'].values)

inds = []

allowed_uids = X_id.loc[X_id['product_uid'].isin(valid_product_uids)]
# For now, always grab first row of valid product uid.
lastUid = 0

for idx, mrow in allowed_uids.iterrows():
    if lastUid == mrow['product_uid']:
        continue

    lastUid = mrow['product_uid']
    inds.append(idx)

test_inds = inds + list(X_id[trainend:].index.values)
train_inds = list(X_id.loc[~X_id.index.isin(test_inds)].index.values)

print("Train: "+str(len(train_inds))+", test: "+str(len(test_inds)))
X_train_train = X_train.iloc[train_inds,2:]
X_train_test = X_train.iloc[test_inds,2:]

y_train_train = y_train[train_inds]
y_train_test = y_train[test_inds]

X_test = X_test.iloc[:,2:]

modellist=['randomforest','extratree','lasso','ridge']
modelresult=modellist(modellist,X_train_train,y_train_train)
result=[]
rmse={}
for model in modelresult:
    result.append(prediction(item,X_train_test))
    rmse[model]==rms(y_train_test,prediction(item,X_train_test))
    return result,rmse
    
    
a



