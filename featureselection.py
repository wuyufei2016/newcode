#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:35:58 2017

@author: XiHUANG
"""

import math
import os
import random
import pandas as pd
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import xgboost as xgb
import operator
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from scipy.sparse import hstack, coo_matrix, csr_matrix
with open('feats.pkl', 'rb') as infile:
    X = pickle.load(infile)

with open('target.pkl', 'rb') as infile:
    y = pickle.load(infile)
X_train = X[:n_train,:]
X_test = X[n_train:,:]

y_train = y.iloc[:n_train,1]

X_id = X_train[:,0:2].todense()
X_id = pd.DataFrame(X_id)
X_id = X_id.astype('int')
X_id.columns = ['id', 'product_uid']
X_train = X.iloc[:n_train,:]
X_test = X.iloc[n_train:,:]

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

y_train = y.iloc[:n_train,1]

X_id = X_train[['id', 'product_uid']]
len(X_id)
trainend = int(len(X_id)*0.885)
counts = X_id[:trainend].groupby(['product_uid']).count()[['id']]

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
X_train_train = X_train[train_inds,:]
X_train_test = X_train[test_inds,:]

y_train_train = y_train.iloc[train_inds]
y_train_test = y_train.iloc[test_inds]
X_train_train = X_train.iloc[train_inds,:]
X_train_test = X_train.iloc[test_inds,:]

X_train_train = X_train_train.fillna(0)
X_train_test= X_train_test.fillna(0)

y_train_train = y_train.iloc[train_inds]
y_train_test = y_train.iloc[test_inds]
dtrain = xgb.DMatrix(X_train_train, label=y_train_train)
dtest = xgb.DMatrix(X_train_test, label=y_train_test)

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
dtrain = xgb.DMatrix(X_train, label=y_train)
evallist  = [(dtrain,'train'), (dtrain,'train')]

# Best score with ~1.4x 
nrounds = round(xgb_fit.best_iteration * 1.4)

xgb_fit_full = xgb.train(param, 
                         dtrain, 
                         nrounds, 
                         evals = evallist, 
                         verbose_eval = 50,
                         learning_rates = learning_rates.tolist())

scores = xgb_fit_full.get_score(importance_type='gain')
sorted_x = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
useful_features = [f for (f, s) in scores.items() if s >= 2.0]
x_lab= ["W2V Euclidean search&title ","W2V Cosine search&title","Cosine hashed 1-Gram search&title","TFIDF Cosine search&title","Search terms count","Search&title overlap","W2V Cosine search&title (mean)","W2V Cosine search&description(mean)","W2V Cosine search&description(best)","Search term length"]
y_lab = [38.30, 28.20, 23.88, 22.8, 20.96, 13.14,11.72,10.24,9.41,9.30]
feature_df = pd.DataFrame()
feature_df["Features"] = x_lab 
feature_df["scores"] = y_lab
import matplotlib.pyplot as plt
#import plotly.plotly as py
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
#tips = sns.load_dataset(feature_df)
sns.barplot(y="Features", x="scores", data=feature_df)
plt.xlabel("Scores")
plt.tight_layout()
plt.savefig('Feature Scores',dpi=400)