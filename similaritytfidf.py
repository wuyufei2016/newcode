#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:30:48 2017

@author: XiHUANG
"""

import time
import os
import numpy as np
import pandas as pd
import pickle as pickle

from collections import Counter
from scipy.sparse import hstack, coo_matrix, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import basic_feature 
start_time = time.time()

import basic_function

def featurematch(test,n):
    result1 = hasher_word(test, n)
    tt = TfidfTransformer()
    result2= tt.fit_transform(result1)
    return result1,result2
    
def cosinesimilarity(test1,test2):
    result1=[cosine_similarity(test1[i], test2[0][0]) for i in range(hash_1_ST.shape[0])]
    return result1
    

start_time = time.time()

ngram = 1

ST_TFIDF = tfidf(df_all['search_term'], ngram)
PT_TFIDF = tfidf(df_all['product_title'], ngram)
PD_TFIDF = tfidf(df_all['product_description'], ngram)
B_TFIDF = tfidf(df_all['brand'], ngram)
V_TFIDF = tfidf(df_all['value'], ngram)

TFIDF_1 = hstack((ST_TFIDF, PT_TFIDF, PD_TFIDF, B_TFIDF, V_TFIDF))

start_time = time.time()


def metchword(ngram):
    ST_TFIDF = tfidf(df_all['search_term'], ngram)
    PT_TFIDF = tfidf(df_all['product_title'], ngram)
    PD_TFIDF = tfidf(df_all['product_description'], ngram)
    B_TFIDF = tfidf(df_all['brand'], ngram)
    V_TFIDF = tfidf(df_all['value'], ngram)
    
    TFIDF_2 = hstack((ST_TFIDF, PT_TFIDF, PD_TFIDF, B_TFIDF, V_TFIDF))
    ngram=1
    hash_1_ST,TFIDF_1_ST = featurematch(df_all['search_term'], ngram)
    hash_1_PT,TFIDF_1_PT = featurematch(df_all['product_title'], ngram)
    hash_1_PD,TFIDF_1_PD = featurematch(df_all['product_description'], ngram)
    hash_1_B,TFIDF_1_B = featurematch(df_all['brand'], ngram)
    hash_1_V,TFIDF_1_V = featurematch(df_all['value'], ngram)
    
    hash_match_1_ST_PT = cosinesimilarity(hash_1_ST, hash_1_PT)
    hash_match_1_ST_PD = cosinesimilarity(hash_1_ST, hash_1_PD)
    hash_match_1_ST_B = cosinesimilarity(hash_1_ST, hash_1_B)
    hash_match_1_ST_V = cosinesimilarity(hash_1_ST, hash_1_V)
    
    TFIDF_match_1_ST_PT = cosinesimilarity(TFIDF_1_ST, TFIDF_1_PT)
    TFIDF_match_1_ST_PD = cosinesimilarity(TFIDF_1_ST, TFIDF_1_PD)
    TFIDF_match_1_ST_B = cosinesimilarity(TFIDF_1_ST, TFIDF_1_B)
    TFIDF_match_1_ST_V = cosinesimilarity(TFIDF_1_ST, TFIDF_1_V)
    feats_match_word_n = pd.concat((df_all['id'],
                                    pd.DataFrame(hash_match_1_ST_PT), 
                                    pd.DataFrame(hash_match_1_ST_PD),
                                    pd.DataFrame(hash_match_1_ST_B),
                                    pd.DataFrame(hash_match_1_ST_V),
                                    pd.DataFrame(TFIDF_match_1_ST_PT), 
                                    pd.DataFrame(TFIDF_match_1_ST_PD),
                                    pd.DataFrame(TFIDF_match_1_ST_B),
                                    pd.DataFrame(TFIDF_match_1_ST_V)),
                                   ignore_index = True,
                                   axis = 1)
    feats_match_word_n.columns = ['id', 'hash_match_1_ST_PT', 'hash_match_1_ST_PD', 'hash_match_1_ST_B', 'hash_match_1_ST_V',
                                  'TFIDF_match_1_ST_PT', 'TFIDF_match_1_ST_PD', 'TFIDF_match_1_ST_B', 'TFIDF_match_1_ST_V']

    return feats_match_word_n 
with open('feats_match_word_1.pkl', 'wb') as outfile:
    pickle.dump(metchword(1), outfile, pickle.HIGHEST_PROTOCOL)
with open('feats_match_word_2.pkl', 'wb') as outfile:
    pickle.dump(metchword(2), outfile, pickle.HIGHEST_PROTOCOL)
group_size = df_all.groupby('product_uid').size().reset_index()
group_size.columns = ['product_uid', 'product_count']
df_all_2 = df_all.merge(group_size, how = 'left', on = 'product_uid')
counts = df_all_2[['id','product_count']]
