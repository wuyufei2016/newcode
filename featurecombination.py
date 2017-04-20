#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:59:31 2017

@author: XiHUANG
"""

import time
import os
import numpy as np
import pandas as pd
import pickle as pickle

with open('w2v_cs_ST_PT_mean_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_cs_ST_PT_mean = pickle.load(infile)
    
with open('w2v_cs_ST_PD_mean_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_cs_ST_PD_mean = pickle.load(infile)
    
with open('w2v_ed_ST_PT_mean_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_ed_ST_PT_mean = pickle.load(infile)
    
with open('w2v_ed_ST_PD_mean_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_ed_ST_PD_mean = pickle.load(infile)
    
w2v_cs_ed = pd.concat((df_all['id'],
                 pd.DataFrame(w2v_cs_ST_PT_mean),
                 pd.DataFrame(w2v_cs_ST_PD_mean), 
                 pd.DataFrame(w2v_ed_ST_PT_mean),
                 pd.DataFrame(w2v_ed_ST_PD_mean)),
                ignore_index = True,
                axis = 1)

w2v_cs_ed.columns = ['id', 'w2v_cs_ST_PT_mean', 'w2v_cs_ST_PD_mean', 'w2v_ed_ST_PT_mean', 'w2v_ed_ST_PD_mean'] 
with open('w2v_cs_ST_PT_ind_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_cs_ST_PT_ind = pickle.load(infile)
    
with open('w2v_cs_ST_PD_ind_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_cs_ST_PD_ind = pickle.load(infile)
    
with open('w2v_ed_ST_PT_ind_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_ed_ST_PT_ind = pickle.load(infile)
    
with open('w2v_ed_ST_PD_ind_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_ed_ST_PD_ind = pickle.load(infile)

with open('w2v_n_cs_ST_PT_ind_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_n_cs_ST_PT_ind = pickle.load(infile)
    
with open('w2v_n_cs_ST_PD_ind_clean_200_mc3_w5.pkl', 'rb') as infile:
    w2v_n_cs_ST_PD_ind = pickle.load(infile)

w2v_cs_ed_ind = pd.concat((df_all['id'],
                 pd.DataFrame(w2v_cs_ST_PT_ind),
                 pd.DataFrame(w2v_cs_ST_PD_ind), 
                 pd.DataFrame(w2v_ed_ST_PT_ind),
                 pd.DataFrame(w2v_ed_ST_PD_ind),
                 pd.DataFrame(w2v_n_cs_ST_PT_ind),
                 pd.DataFrame(w2v_n_cs_ST_PD_ind),),
                ignore_index = True,
                axis = 1)

w2v_cs_ed_ind.columns = ['id', 'w2v_cs_ST_PT_ind', 'w2v_cs_ST_PD_ind', 'w2v_ed_ST_PT_ind', 'w2v_ed_ST_PD_ind',
                         'w2v_n_cs_ST_PT_ind', 'w2v_n_cs_ST_PD_ind'] 
target_feats = df_all[['id', 'product_uid', 'relevance']].merge(feats_match_char, how = 'left', on = 'id')
target_feats = target_feats.merge(feats_match_word_1, how = 'left', on = 'id')
target_feats = target_feats.merge(feats_match_word_2, how = 'left', on = 'id')
target_feats = target_feats.merge(feats_olap, how = 'left', on = 'id')
target_feats = target_feats.merge(feats_tf, how = 'left', on = 'id')
target_feats = target_feats.merge(feats_l, how = 'left', on = 'id')
#target_feats = target_feats.merge(feats_rl, how = 'left', on = 'id')
target_feats = target_feats.merge(counts, how = 'left', on = 'id')
target_feats = target_feats.merge(ST_counts, how = 'left', on = 'id')
target_feats = target_feats.merge(w2v, how = 'left', on = 'id')
target_feats = target_feats.merge(w2v_cs_ed, how = 'left', on = 'id')
target_feats = target_feats.merge(w2v_cs_ed_ind, how = 'left', on = 'id')
#target_feats = target_feats.merge(d2v_cs_ed, how = 'left', on = 'id')
#target_feats = target_feats.merge(d2v, how = 'left', on = 'id')

target = target_feats[['id', 'relevance']]

feats = target_feats.drop(['relevance'], axis = 1)

'''
#Remove atrribute columns:
rem_col = feats.columns.to_series().str.contains('_PT')
rem_col[rem_col.isnull()] = False
feats = feats.drop(list(feats.columns[list(rem_col)]), axis = 1)
'''

'''
feats_sparse = coo_matrix(feats.values, dtype= 'float64')
feats_sparse = hstack((feats_sparse, TFIDF_1))
feats_sparse = hstack((feats_sparse, TFIDF_2))
feats_sparse = csr_matrix(feats_sparse)  #can't index coo matrics 
with open('feats_sparse.pkl', 'wb') as outfile:
    pickle.dump(feats_sparse, outfile, pickle.HIGHEST_PROTOCOL)
'''

with open('target.pkl', 'wb') as outfile:
    pickle.dump(target, outfile, pickle.HIGHEST_PROTOCOL)
    
with open('feats.pkl', 'wb') as outfile:
    pickle.dump(feats, outfile, pickle.HIGHEST_PROTOCOL)