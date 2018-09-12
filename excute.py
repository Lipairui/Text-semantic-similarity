#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals
from semantic_vectors_generator import *
from calculate_similarity import *
from data_preprocess import *
from config import *
import pandas as pd
import sys
import codecs
import argparse
'''
command format:
    excute.py txt1_path txt2_path save_path -en/cn 
txt format:
    utf-8 encoding required
    one sentence/paragraph per line (split with '\n')
'''
# user interaction
parser = argparse.ArgumentParser(description='Calculate semantic similarity of two texts.  Models include word2vec, tfidf, lda, lsi. User can combine more than one of the models by change config.py setting.')
parser.add_argument("text1_path",help='the path of text1(.txt format; utf-8 encoding required; one sentence/paragraph per line)')
parser.add_argument("text2_path",help='the path of text2(.txt format; utf-8 encoding required; one sentence/paragraph per line)')
parser.add_argument("res_path",help='the path of result(.csv format)')
parser.add_argument('-l','--language',help='language of the texts: en for English; cn for Chinese',choices=['en','cn'])
args = parser.parse_args()

# read data
data1 = codecs.open(args.text1_path,'r',encoding='utf-8')
data2 = codecs.open(args.text2_path,'r',encoding='utf-8')
data1 = data1.read().split('\n')
data2 = data2.read().split('\n')

# preprocess data
if args.language=='en':
    pdata1 = preprocess_data_en(data1)
    pdata2 = preprocess_data_en(data2)
elif args.language=='cn':
    pdata1 = preprocess_data_cn(data1)
    pdata2 = preprocess_data_cn(data2)
pdata = pdata1+pdata2
# get semantic features vectors

features = generate_semantic_features(pdata,config)
features1 = features[:len(pdata1)]
features2 = features[len(pdata1):].reset_index(drop=True)

# calculate semantic similarity
sims = calculate_semantic_similarity(features1,features2)

# save result as .csv
res = pd.DataFrame(columns=['text1','text2','similarity'])
res.text1 = data1
res.text2 = data2
res.similarity = sims
res.to_csv(args.res_path,index=0)
LogInfo(' Save result successfully!')