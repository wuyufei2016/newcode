#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:57:49 2017

@author: XiHUANG
"""

import time
import os
import numpy as np
import pandas as pd
import pickle as pickle
import gensim
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import euclidean, cosine
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, LabeledSentence, TaggedDocument
from random import shuffle
from itertools import chain

with open('df_all_cleaner.pkl', 'rb') as infile:
    df_all = pickle.load(infile)
df = df_all['product_title'] + ' ' + df_all['product_description'] #+ ' ' + df_all['value']

df = pd.DataFrame(df)
df.columns = ['product_info']

df = pd.concat([df_all['product_uid'], df], axis=1)
df = df.drop_duplicates()

docs = [TaggedDocument(words = df['product_info'].iloc[i_row].split(), tags = ["row" + str(i_row)])
        for i_row in range(len(df))]
model = Doc2Vec(size = 200, min_count=3, window=5, workers=4)
model.build_vocab(docs)

for epoch in range(10):
    shuffle(docs)
    model.train(docs)

model.init_sims(replace=True)
with open('d2v_clean_stem_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(model, outfile, pickle.HIGHEST_PROTOCOL)
with open('d2v_clean_stem_200_mc3_w5.pkl', 'rb') as infile:
    model = pickle.load(infile)
docvec = [model.docvecs["row" + str(df_all['product_uid'].iloc[i] - 100001)] for i in range(len(df_all['product_uid']))]

with open('d2v_docvec_clean_stem_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(docvec, outfile, pickle.HIGHEST_PROTOCOL)
n_row = len(df_all['search_term'])

vocab_set = set(model.vocab.keys())

ST_vec = np.zeros((n_row, model.syn0.shape[1]))

for i_row in range(n_row):
    if i_row % 10000 == 0:
        print(i_row)
    
    ST_set = set(df_all['search_term'][i_row].split())
    olap = vocab_set & ST_set
    if olap:
        ST_vec[i_row,:] = model[olap].mean(axis=0)
    
with open('d2v_STvec_clean_stem_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(ST_vec, outfile, pickle.HIGHEST_PROTOCOL)
with open('d2v_STvec_clean_stem_200_mc3_w5.pkl', 'rb') as infile:
    ST_vec = pickle.load(infile)

with open('d2v_docvec_clean_stem_200_mc3_w5.pkl', 'rb') as infile:
    docvec = pickle.load(infile)

d2v_cs_ST_docvec = [cosine(ST_vec[i,:], docvec[i]) for i in range(len(ST_vec))]
d2v_ed_ST_docvec = [cosine(ST_vec[i,:], docvec[i]) for i in range(len(ST_vec))]

with open('d2v_cs_ST_docvec_mean_clean_stem_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(d2v_cs_ST_docvec, outfile, pickle.HIGHEST_PROTOCOL)
    
with open('d2v_ed_ST_docvec_mean_clean_stem_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(d2v_ed_ST_docvec, outfile, pickle.HIGHEST_PROTOCOL)