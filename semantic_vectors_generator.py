#!/usr/bin/env: python3
# -*- coding: utf-8 -*-
import time
import re
import jieba
import codecs
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim, logging
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
from gensim.models.lsimodel import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from matplotlib import pyplot
from data_preprocess import *

def LogInfo(stri):
    '''
     Funciton: 
         print log information
     Input:
         stri: string
     Output: 
         print time+string
     '''
    
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)


def getColName(colNum, stri):
    '''
     Funciton: 
         generate columns names
     Input: 
         colNum: number of columns
         stri: string
     Output:
         list of columns names   
    '''
    LogInfo(' '+str(colNum)+','+stri)
    colName = []
    for i in range(colNum):
        colName.append(stri + str(i))
    return colName

def get_pretrained_w2vfeatures(documents,model_path):
    '''
     Funciton:
         generate word2vec features with the pretrained word2vec gensim model
     Input:
         documents: list of preprocessed sentences
         model_path: path of the pretrained word2vec gensim model
     Output:
         word2vec features(DataFrame format)
    '''
    # reconstruct corpus according to word frequency 
#     LogInfo(' Reconstruct corpus...')
    min_word_freq = 2  
    texts = [[word for word in document.split(' ')] for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] >= min_word_freq] for text in texts]

    # get pretrained word2vec model
#     LogInfo(' Get pretrained w2vmodel...')
    model = gensim.models.Word2Vec.load(model_path)
 
    # generate w2vFeatures
    LogInfo(' Generate word2vec features...')
    topicNum = 400
    w2vFeature = np.zeros((len(texts), topicNum))
    w2vFeatureAvg = np.zeros((len(texts), topicNum))
    i = 0
    error = 0  
    for line in texts:
        num = 0
        for word in line:
            num += 1
            try:
                vec = model[word]
            except:
                print('Error: '+word) 
                error += 1               
                vec = np.zeros(topicNum)  
            w2vFeature[i, :] += vec

        w2vFeatureAvg[i,:] = w2vFeature[i,:]/num
        i += 1   
#     print('Total errors: ',error)  
    colName = getColName(topicNum, "vecT")
    w2vFeature = pd.DataFrame(w2vFeatureAvg, columns = colName)
    return w2vFeature


def get_selftrained_w2vfeatures(documents,topicNum):
    '''
     Funciton:
         generate word2vec features by training word2vec model
     Input:
         documents: list of preprocessed sentences
         topicNum: output vector dimension
     Output:
         word2vec features(DataFrame format)
    '''
    # reconstruct corpus according to word frequency  
#     LogInfo(' Reconstruct corpus...')
    min_word_freq = 1    
    texts = [[word for word in document.split(' ')] for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] >= min_word_freq] for text in texts]

    # train word2vec model according to the corpus
#     LogInfo(' Train word2vec Model...')
    w2vmodel = Word2Vec(texts, size=topicNum, window=5, iter = 15, min_count=min_word_freq, workers=12, seed = 12)#, sample = 1e-5, iter = 10,seed = 1)
#     path = '../model/'+str(topicNum)+'w2vModel.m'
#     w2vmodel.save(path)

    # generate w2vFeatures
    LogInfo(' Generate word2vec features...')
    w2vFeature = np.zeros((len(texts), topicNum))
    w2vFeatureAvg = np.zeros((len(texts), topicNum))
    i = 0
    for line in texts:
        num = 0
        for word in line:
            num += 1
            vec = w2vmodel[word]
            w2vFeature[i, :] += vec
        w2vFeatureAvg[i,:] = w2vFeature[i,:]/num
        i += 1 
    colName = getColName(topicNum, "vecT")
    w2vFeature = pd.DataFrame(w2vFeatureAvg, columns = colName)
    return w2vFeature

def getLsiFeature(documents, topicNum):
    '''
     Funciton:
         generate lsi features by training lsi model
     Input:
         documents: list of preprocessed sentences
         topicNum: output vector dimension
     Output:
         lsi features(DataFrame format)
    '''
    # get corpus
#     LogInfo(' Get corpus...')
    texts = [[word for word in document.split(' ')] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpusD = [dictionary.doc2bow(text) for text in texts]
    
    # train lsi model
#     LogInfo(' Train LSI model...')
    tfidf = TfidfModel(corpusD)
    corpus_tfidf = tfidf[corpusD]
    model = LsiModel(corpusD, num_topics=topicNum, chunksize=8000, extra_samples = 100)#, distributed=True)#, sample = 1e-5, iter = 10,seed = 1)

    # generate lsi features
    LogInfo(' Generate LSI features...')
    lsiFeature = np.zeros((len(texts), topicNum))
    i = 0
    for doc in corpusD:
        topic = model[doc]
        for t in topic:
             lsiFeature[i, t[0]] = round(t[1],5)
        i = i + 1
    colName = getColName(topicNum, "qlsi")
    lsiFeature = pd.DataFrame(lsiFeature, columns = colName)
    return lsiFeature


def getLdaFeature(documents, topicNum):
    '''
     Funciton:
         generate lda features by training lda model
     Input:
         documents: list of preprocessed sentences
         topicNum: output vector dimension
     Output:
         lda features(DataFrame format)
    '''
    # get corpus
#     LogInfo(' Get corpus...')
    texts = [[word for word in document.split(' ')] for document in documents]
    dictionary = corpora.Dictionary(texts)    
    corpusD = [dictionary.doc2bow(text) for text in texts]

    # train lda model
#     LogInfo(' Train LDA model...')
    tfidf = TfidfModel(corpusD)
    corpus_tfidf = tfidf[corpusD]
#     ldaModel = gensim.models.ldamulticore.LdaMulticore(corpus_tfidf, workers = 8, num_topics=topicNum, chunksize=8000, passes=10, random_state = 12)
    ldaModel = LdaModel(corpus_tfidf, num_topics=topicNum, chunksize=8000, passes=10, random_state = 12)
    # generate lda features
    LogInfo(' Generate LDA features...')
    ldaFeature = np.zeros((len(texts), topicNum))
    i = 0
    for doc in corpus_tfidf:
        topic = ldaModel.get_document_topics(doc, minimum_probability = 0.01)
        for t in topic:
             ldaFeature[i, t[0]] = round(t[1],5)
        i = i + 1
    colName = getColName(topicNum, "qlda")
    ldaFeature = pd.DataFrame(ldaFeature, columns = colName)
    return ldaFeature

def get_tfidf_feature(documents):
    '''
     Funciton:
         generate tfidf features 
     Input:
         data: list of preprocessed sentences    
     Output:
         tfidf features(DataFrame format)
    '''   
    LogInfo(' Generate TFIDF features...')
    tfidf = TfidfVectorizer()
    res = tfidf.fit_transform(documents).toarray()
    dim = len(tfidf.get_feature_names())
    colName = getColName(dim, "tfidf")
    tfidf_features = pd.DataFrame(res,columns = colName)
    return tfidf_features

def generate_semantic_features(data,config):
    '''
     Funciton:
         generate all semantic features according to config
     Input:
         data: list of preprocessed sentences    
         config: model setting (dict)
     Output:
         semantic features (DataFrame format)
     '''   
    features = []
    
    # word2vec
    if config['word2vec']==0:
        # default model setting
        dim = 800 
        w2vFeature = get_selftrained_w2vfeatures(data,dim)
        features.append(w2vFeature)
    elif config['word2vec']>0:
        # user's model setting
        dim = config['word2vec']
        w2vFeature = get_selftrained_w2vfeatures(data,dim)
        features.append(w2vFeature)
    
    # tfidf
    if config['tfidf']==0:
        tfidfFeature = get_tfidf_feature(data)
        features.append(tfidfFeature)
    
    # lda
    if config['lda']==0:
        # default model setting
        dim = 10
        ldaFeature = getLdaFeature(data,dim)
        features.append(ldaFeature)
    elif config['lda']>0:
        # user's model setting
        dim = config['lda']
        ldaFeature = getLdaFeature(data,dim)
        features.append(ldaFeature)        
    
    # lsi
    if config['lsi']==0:
        # default model setting
        dim = 500
        lsiFeature = getLsiFeature(data,dim)
        features.append(lsiFeature)
    elif config['lsi']>0:
        # user's model setting
        dim = config['lsi']
        lsiFeature = getLsiFeature(data,dim)
        features.append(lsiFeature)
    
    features = pd.concat(features,axis=1)
    return features
        
    


