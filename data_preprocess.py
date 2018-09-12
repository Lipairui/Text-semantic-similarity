#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import jieba
import jieba.posseg
import jieba.analyse
import codecs
import nltk
from nltk.stem import SnowballStemmer  

def preprocess_data_en(data):
    '''
    Function: preprocess data in English including cleaning, tokenzing, stemming...
    Input: list of strings (e.g. [sentence1,sentence2...])
    Output: list of preprocessed strings (e.g. [preprocessed_sentence1,preprocessed_sentence2...])

    '''
    # clean data and tokenize
    pattern = r"""(?x)                   # set flag to allow verbose regexps 
                  (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A. 
                  |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages 
                  |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe 
                  |\.\.\.                # ellipsis 
                  |(?:[.,;"'?():-_`])    # special characters with meanings 
                """ 
    data = [nltk.regexp_tokenize(sentence, pattern) for sentence in data]

    # move stopwords & lower words
    stopworddic = set([w.strip() for w in codecs.open('english_stopwords.txt', 'r',encoding='utf-8').readlines()])
    clean_data = []
    for sentence in data:
        clean_data += [[word.lower() for word in sentence if word.lower() not in stopworddic]]
    data = clean_data

    # stemming
    snowball_stemmer = SnowballStemmer('english') 
    stem_data = []
    for sentence in data:
        stem_data += [[snowball_stemmer.stem(word) for word in sentence]] 
    data = stem_data

    # join words to sentence
    data = [' '.join(sentence) for sentence in data]

    return data

def preprocess_data_cn(data):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: list of strings (e.g. [sentence1,sentence2...])
    Output: list of preprocessed strings (e.g. [preprocessed_sentence1,preprocessed_sentence2...])

    '''    
    
    # clean data
    data = [re.sub(u"[^\u4E00-\u9FFF]", "", sentence) for sentence in data] # delete all non-chinese characters
    
    # tokenize and move stopwords 
    stopworddic = [w.strip() for w in codecs.open('chinese_stopwords.txt', 'r',encoding='utf-8').readlines()]
    result = []
    pos = ['zg','e','y','o','ul','ud','uj','z'] # 定义需要过滤的词性
    # zg:哦 e:嗯 y:啦 o:哈哈 ul:了 r:他，你，哪儿，哪里 ug:过 z:咋啦
    for text in data:
        words = []
        seg = jieba.posseg.cut(text)  # 分词
        for i in seg:   
            if i.flag not in pos and i.word not in stopworddic :  # 去停用词 + 词性筛选
                words.append(i.word)
        words = ' '.join(words) # join words to sentence
        result.append(words)
    data = result
    
    return data



