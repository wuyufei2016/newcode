#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:55:07 2017

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

assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
with open('df_all_cleaner.pkl', 'rb') as infile:
    df_all = pickle.load(infile)
model = Word2Vec.load_word2vec_format('googlew2v.bin.gz', binary=True)

df = df_all['product_title'] + ' ' + df_all['product_description'] #+ ' ' + df_all['value']

df = pd.DataFrame(df)
df.columns = ['product_info']

df = pd.concat([df_all['product_uid'], df], axis=1)
df = df.drop_duplicates()

sentences = [df['product_info'].iloc[i].split() for i in range(len(df['product_info']))]

model = Word2Vec(sentences, size = 200, min_count = 3, window = 5, workers = 2) 
model.init_sims(replace=True)

with open('w2v_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(model, outfile, pickle.HIGHEST_PROTOCOL)  
n_row = len(df_all['search_term'])

vocab_set = set(model.vocab.keys())

vec_ST_mean = np.zeros((n_row, model.syn0.shape[1]))
vec_PT_mean = np.zeros((n_row, model.syn0.shape[1]))
vec_PD_mean = np.zeros((n_row, model.syn0.shape[1]))

for i_row in range(n_row):
    if i_row % 10000 == 0:
        print(i_row)
    
    ST_set = set(df_all['search_term'][i_row].split())
    olap = vocab_set & ST_set
    if olap:
        vec_ST_mean[i_row,:] = model[olap].mean(axis=0)
    
    PT_set = set(df_all['product_title'][i_row].split())
    olap = vocab_set & PT_set
    if olap:
        vec_PT_mean[i_row,:] = model[olap].mean(axis=0)
    
    PD_set = set(df_all['product_description'][i_row].split())
    olap = vocab_set & PD_set
    if olap:
        vec_PD_mean[i_row,:] = model[olap].mean(axis=0)

with open('w2v_ST_mean_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(vec_ST_mean, outfile, pickle.HIGHEST_PROTOCOL)
    
with open('w2v_PT_mean_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(vec_PD_mean, outfile, pickle.HIGHEST_PROTOCOL)
    
with open('w2v_PD_mean_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(vec_PD_mean, outfile, pickle.HIGHEST_PROTOCOL)
with open('w2v_ST_mean_cleaner_200_mc3_w5.pkl', 'rb') as infile:
    vec_ST_mean = pickle.load(infile)

with open('w2v_PT_mean_cleaner_200_mc3_w5.pkl', 'rb') as infile:
    vec_PT_mean = pickle.load(infile)

with open('w2v_PD_mean_cleaner_200_mc3_w5.pkl', 'rb') as infile:
    vec_PD_mean = pickle.load(infile)

w2v_cs_ST_PT_mean = [cosine(vec_ST_mean[i,:], vec_PT_mean[i,:]) for i in range(len(vec_ST_mean))]
w2v_cs_ST_PD_mean = [cosine(vec_ST_mean[i,:], vec_PD_mean[i,:]) for i in range(len(vec_ST_mean))]
w2v_ed_ST_PT_mean = [euclidean(vec_ST_mean[i,:], vec_PT_mean[i,:]) for i in range(len(vec_ST_mean))]
w2v_ed_ST_PD_mean = [euclidean(vec_ST_mean[i,:], vec_PD_mean[i,:]) for i in range(len(vec_ST_mean))]

with open('w2v_cs_ST_PT_mean_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_cs_ST_PT_mean, outfile, pickle.HIGHEST_PROTOCOL)

with open('w2v_cs_ST_PD_mean_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_cs_ST_PD_mean, outfile, pickle.HIGHEST_PROTOCOL)

with open('w2v_ed_ST_PT_mean_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_ed_ST_PT_mean, outfile, pickle.HIGHEST_PROTOCOL)
    
with open('w2v_ed_ST_PD_mean_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_ed_ST_PD_mean, outfile, pickle.HIGHEST_PROTOCOL)
n_row = len(df_all['search_term'])

vocab_set = set(model.vocab.keys())

w2v_cs_ST_PT_ind = np.zeros(n_row)
w2v_cs_ST_PD_ind = np.zeros(n_row)
w2v_ed_ST_PT_ind = np.zeros(n_row)
w2v_ed_ST_PD_ind = np.zeros(n_row)

w2v_n_cs_ST_PT_ind = np.zeros(n_row)
w2v_n_cs_ST_PD_ind = np.zeros(n_row)

for i_row in range(n_row):
    if i_row % 10000 == 0:
        print(i_row)
    
    ST_set = set(df_all['search_term'][i_row].split())
    olap = vocab_set & ST_set
    if olap:
        vec_ST = model[olap]
    
    PT_set = set(df_all['product_title'][i_row].split())
    olap = vocab_set & PT_set
    if olap:
        vec_PT = model[olap]
    
    PD_set = set(df_all['product_description'][i_row].split())
    olap = vocab_set & PD_set
    if olap:
        vec_PD = model[olap]
    
    if vec_ST.any() and vec_PT.any():
        cs_vec = cosine_similarity(vec_ST, vec_PT)
        ed_vec = euclidean_distances(vec_ST, vec_PT)
        
        w2v_cs_ST_PT_ind[i_row] = np.sum(cs_vec) / cs_vec.size
        w2v_ed_ST_PT_ind[i_row] = np.sum(ed_vec) / ed_vec.size
        
        #Discard exact matches between words and count pairs with relatively large cs:
        cs_vec[cs_vec > .99] = 0
        w2v_n_cs_ST_PT_ind[i_row] = np.sum(cs_vec > 0.4)
        
    if vec_ST.any() and vec_PD.any():
        cs_vec = cosine_similarity(vec_ST, vec_PD)
        ed_vec = euclidean_distances(vec_ST, vec_PD)
        
        w2v_cs_ST_PD_ind[i_row] = np.sum(cs_vec) / cs_vec.size
        w2v_ed_ST_PD_ind[i_row] = np.sum(ed_vec) / ed_vec.size
        
        #Discard exact matches between words and count pairs with relatively large cs:
        cs_vec[cs_vec > .99] = 0
        w2v_n_cs_ST_PT_ind[i_row] = np.sum(cs_vec > 0.4)
        
        
with open('w2v_cs_ST_PT_ind_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_cs_ST_PT_ind, outfile, pickle.HIGHEST_PROTOCOL)

with open('w2v_cs_ST_PD_ind_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_cs_ST_PD_ind, outfile, pickle.HIGHEST_PROTOCOL)

with open('w2v_ed_ST_PT_ind_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_ed_ST_PT_ind, outfile, pickle.HIGHEST_PROTOCOL)

with open('w2v_ed_ST_PD_ind_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_ed_ST_PD_ind, outfile, pickle.HIGHEST_PROTOCOL)

with open('w2v_n_cs_ST_PT_ind_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_n_cs_ST_PT_ind, outfile, pickle.HIGHEST_PROTOCOL)

with open('w2v_n_cs_ST_PD_ind_cleaner_200_mc3_w5.pkl', 'wb') as outfile:
    pickle.dump(w2v_n_cs_ST_PD_ind, outfile, pickle.HIGHEST_PROTOCOL)

