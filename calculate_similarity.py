#!/usr/bin/env: python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from data_preprocess import *
from semantic_vectors_generator import *
from sklearn.metrics import accuracy_score
#计算语义相似度
def calculate_semantic_similarity(features1,features2):
    '''
     Funciton:
         calculate two datasets' semantic similarity 
     Input:
         features1: dataset1's semantic features (DataFrame format)
         features2: dataset2's semantic features (DataFrame format)
     Output:
         list of semantic similarity of texts
     '''
    
    size = len(features1)
    features1 = features1.values
    features2 = features2.values
    vectors_sim = []
    for i in range(size):
        vec1 = features1[i]
        vec2 = features2[i]
        # 余弦相似度
        cos = np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        sim = 1-np.arccos(cos)/np.pi
        vectors_sim.append(sim)
    return vectors_sim


    
    
    