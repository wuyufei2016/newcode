#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:06:34 2017

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

def tfidf(docs, ngram):
    t = TfidfVectorizer(decode_error = 'ignore',
                        analyzer = 'word',
                        stop_words = 'english',
                        ngram_range = (ngram, ngram), 
                        max_df = 0.8, 
                        min_df = 0, 
                        max_features = 20000)
    tfidf_mat = t.fit_transform(docs)
    return tfidf_mat

def sum_count(s, counts):
    tf = 0
    for i in range(len(s.split())):
        tf = tf + counts[s.split()[i]]
    return tf
    
def safe_divide(num, den, val = 0.0):
    if den != 0.0:
        val = float(num) / den
    return val

def hasher_char(s):
    h = HashingVectorizer(decode_error = 'ignore',
                           analyzer = 'char',
                           ngram_range = (2,4),
                           stop_words = 'english',
                           n_features = 2 ** 18,
                           non_negative = True,
                           norm = None)
    s = s.replace(" ", "")
    hash_vec = h.transform([s]).toarray()
    return hash_vec

def hasher_word(s, ngram):
    h = HashingVectorizer(decode_error = 'ignore',
                           analyzer = 'word',
                           ngram_range = (ngram, ngram),
                           stop_words = 'english',
                           n_features = 2 ** 18,
                           non_negative = True,
                           norm = None)
    hash_vec = h.fit_transform(s)
    return hash_vec
    
def olap(s1, s2):
    l = len(set(s1.split()) & set(s2.split()))
    return l
with open('df_all_clean.pkl', 'rb') as infile:
    df_all = pickle.load(infile)

start_time = time.time()

match_char_ST_PT = [cosine_similarity(hasher_char(df_all['search_term'][i]), hasher_char(df_all['product_title'][i]))[0][0]
                    for i in range(len(df_all['search_term']))]
print("PT done")
match_char_ST_PD = [cosine_similarity(hasher_char(df_all['search_term'][i]), hasher_char(df_all['product_description'][i]))[0][0]
                    for i in range(len(df_all['search_term']))]
print("PD done")
match_char_ST_B = [cosine_similarity(hasher_char(df_all['search_term'][i]), hasher_char(df_all['brand'][i]))[0][0]
                   for i in range(len(df_all['search_term']))]
print("B done")
match_char_ST_V = [cosine_similarity(hasher_char(df_all['search_term'][i]), hasher_char(df_all['value'][i]))[0][0]
                   for i in range(len(df_all['search_term']))]

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
