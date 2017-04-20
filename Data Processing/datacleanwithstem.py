#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:04:17 2017

@author: XiHUANG
"""

import time
import os
import re
import numpy as np
import pandas as pd
import pickle as pickle

#from nltk.stem.porter import *
#stemmer = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
stemmer = SnowballStemmer('english')
  
from sklearn.feature_extraction.text import HashingVectorizer

def cleaner_str(s):
    cleaner = HashingVectorizer(decode_error = 'ignore',
                           analyzer = 'word',
                           ngram_range = (1,1),
                           stop_words = 'english')
    c = cleaner.build_analyzer()
    s = (" ").join(c(s))
    return s

def stem_str(s):
    if isinstance(s, str):
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s
    else:
        return "null"
        
start_time = time.time()

df_all['search_term'] = [cleaner_str(x) for x in df_all['search_term']]
df_all['product_title'] = [cleaner_str(x) for x in df_all['product_title']]
df_all['product_description'] = [cleaner_str(x) for x in df_all['product_description']]
df_all['brand'] = [cleaner_str(x) for x in df_all['brand']]
df_all['value'] = [cleaner_str(x) for x in df_all['value']]

df_all = df_all.fillna('')

with open('df_all_cleaner.pkl', 'wb') as outfile:
    pickle.dump(df_all, outfile, pickle.HIGHEST_PROTOCOL)

print("--- Cleaning of text: %s minutes ---" % round(((time.time() - start_time)/60),2))

start_time = time.time()

df_all['search_term'] = [stem_str(x) for x in df_all['search_term']]
df_all['product_title'] = [stem_str(x) for x in df_all['product_title']]
df_all['product_description'] = [stem_str(x) for x in df_all['product_description']]
df_all['brand'] = [stem_str(x) for x in df_all['brand']]
df_all['value'] = [stem_str(x) for x in df_all['value']]

df_all = df_all.fillna('')

with open('df_all_cleaner_stem.pkl', 'wb') as outfile:
    pickle.dump(df_all, outfile, pickle.HIGHEST_PROTOCOL)

print("--- Stemming of text: %s minutes ---" % round(((time.time() - start_time)/60),2))