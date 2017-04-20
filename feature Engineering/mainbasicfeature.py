#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:09:47 2017

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
    

def match_char(test2):
    result=[cosine_similarity(hasher_char(df_all['search_term'][i]), hasher_char(test2)[0][0]
                    for i in range(len(test2)))]
    return result
def olapdf(test2):
    result=[olap(df_all['search_term'][i],test2[0][0] for i in range(len(test2)))]
    return result
def count(test):
    st = ' '.join(test)
    st = st.split()
    counts = Counter(st)
    result = [sum_count(x, counts) for x in test]
    return result
match_char_ST_PT = match_char(df_all['product_title'])
                   
print("PT done")
match_char_ST_PD = match_char(df_all['product_description'])
print("PD done")
match_char_ST_B = match_char_ST_PD = match_char(df_all['brand'])
print("B done")
match_char_ST_V = match_char_ST_PD = match_char(df_all['value'])

feats_match_char = pd.concat((df_all['id'],
                              pd.DataFrame(match_char_ST_PT), 
                              pd.DataFrame(match_char_ST_PD),
                              pd.DataFrame(match_char_ST_B),
                              pd.DataFrame(match_char_ST_V)),
                             ignore_index = True,
                             axis = 1)
feats_match_char.columns = ['id', 'match_char_ST_PT', 'match_char_ST_PD', 'match_char_ST_B', 'match_char_ST_V']

with open('feats_match_char.pkl', 'wb') as outfile:
    pickle.dump(feats_match_char, outfile, pickle.HIGHEST_PROTOCOL)
    
print("--- Calculating cosine between hashing vectors: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()
olap_ST_PT = olapdf(df_all['product_title'])
                   

olap_ST_PD = olapdf(df_all['product_description'])

olap_ST_V = olapdf(df_all['brand'])

match_char_ST_V = match_char_ST_PD = match_char(df_all['value'])
feats_olap = pd.concat((df_all['id'],
                        pd.DataFrame(olap_ST_PT),
                        pd.DataFrame(olap_ST_PD),
                        pd.DataFrame(olap_ST_B),
                        pd.DataFrame(olap_ST_V)),
                       ignore_index = True,
                       axis = 1)

feats_olap.columns = ['id', 'olap_ST_PT', 'olap_ST_PD', 'olap_ST_B', 'olap_ST_V']

with open('feats_olap.pkl', 'wb') as outfile:
    pickle.dump(feats_olap, outfile, pickle.HIGHEST_PROTOCOL)
    
print("--- Calculating overlap: %s minutes ---" % round(((time.time() - start_time)/60),2))

tf_ST=count(df_all['search_term'])
tf_PT=count(df_all['product_title'])
tf_PD=count(df_all['product_description'])
tf_B=count(df_all['brand'])
tf_V=count(df_all['value'])
feats_tf.columns = ['id', 'tf_ST', 'tf_PT', 'tf_PD', 'tf_B', 'tf_V']

with open('feats_tf.pkl', 'wb') as outfile:
    pickle.dump(feats_tf, outfile, pickle.HIGHEST_PROTOCOL)

l_ST = [len(x.split()) for x in df_all['search_term']]
l_PT = [len(x.split()) for x in df_all['product_title']]
l_PD = [len(x.split()) for x in df_all['product_description']]
l_B = [len(x.split()) for x in df_all['brand']]
l_V = [len(x.split()) for x in df_all['value']]

feats_l = pd.concat((df_all['id'],
                     pd.DataFrame(l_ST),
                     pd.DataFrame(l_PT),
                     pd.DataFrame(l_PD),
                     pd.DataFrame(l_B),
                     pd.DataFrame(l_V)),
                    ignore_index = True,
                    axis = 1)

feats_l.columns = ['id', 'l_ST', 'l_PT', 'l_PD', 'l_B', 'l_V']

with open('feats_l.pkl', 'wb') as outfile:
    pickle.dump(feats_l, outfile, pickle.HIGHEST_PROTOCOL)

l_char_ST = [len(x.replace(" ", "")) for x in df_all['search_term']]
l_char_PT = [len(x.replace(" ", "")) for x in df_all['product_title']]
l_char_PD = [len(x.replace(" ", "")) for x in df_all['product_description']]
l_char_B = [len(x.replace(" ", "")) for x in df_all['brand']]
l_char_V = [len(x.replace(" ", "")) for x in df_all['value']]

feats_l_char = pd.concat((df_all['id'],
                          pd.DataFrame(l_char_ST),
                          pd.DataFrame(l_char_PT),
                          pd.DataFrame(l_char_PD),
                          pd.DataFrame(l_char_B),
                          pd.DataFrame(l_char_V)),
                         ignore_index = True,
                         axis = 1)

feats_l_char.columns = ['id', 'l_char_ST', 'l_char_PT', 'l_char_PD', 'l_char_B', 'l_char_V']

with open('feats_l_char.pkl', 'wb') as outfile:
    pickle.dump(feats_l_char, outfile, pickle.HIGHEST_PROTOCOL)

rl_PT = [safe_divide(len(df_all['search_term'][i].split()), len(df_all['product_title'][i].split()))
         for i in range(len(df_all['search_term']))]
rl_PD = [safe_divide(len(df_all['search_term'][i].split()), len(df_all['product_description'][i].split()))
         for i in range(len(df_all['search_term']))]
rl_B = [safe_divide(len(df_all['search_term'][i].split()), len(df_all['brand'][i].split()))
        for i in range(len(df_all['search_term']))]
rl_V = [safe_divide(len(df_all['search_term'][i].split()), len(df_all['value'][i].split()))
        for i in range(len(df_all['search_term']))]

feats_rl = pd.concat((df_all['id'],
                      pd.DataFrame(rl_PT),
                      pd.DataFrame(rl_PD),
                      pd.DataFrame(rl_B),
                      pd.DataFrame(rl_V)),
                     ignore_index = True,
                     axis = 1)

feats_rl.columns = ['id', 'rl_PT', 'rl_PD', 'rl_B', 'rl_V']

with open('feats_rl.pkl', 'wb') as outfile:
    pickle.dump(feats_rl, outfile, pickle.HIGHEST_PROTOCOL)

