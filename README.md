# text-semantic-similarity
Calculate semantic similarity of each pair of sentences/paragraphs in two texts (English or Chinese).  
Models include word2vec, tfidf, lda, lsi.

## Dependencies
Python 3.6.5   
pandas, numpy, jieba, nltk, gensim, re, sklearn, codecs, time

## Implementation
Since semantic similarity labeling is difficult and time consuming, unsupervised semantic similarity caculating is useful.   
I choose four models to calculate semantic vectors of texts, then utilize cosine distance to calculate similarity.

## How to use?
Prepare your data: pairs of sentence/paragraphs in two txt files
Command: excute.py text1_path text2_path res_path -l en/cn    
You can run excute.py with -h to get information about arguments details.   
usage: excute.py [-h] [-l {en,cn}] text1_path text2_path res_path   
positional arguments:   
text1_path            the path of text1(.txt format; utf-8 encoding required; one sentence/paragraph per line)    
text2_path            the path of text2(.txt format; utf-8 encoding required; one sentence/paragraph per line)    
res_path              the path of result(.csv format)    
optional arguments:    
-h, --help            show this help message and exit   
-l {en,cn}, --language {en,cn}   
                      language of the texts: en for English; cn for Chinese
                        
## How to change model setting?
I implement four models to calculate unsupervised semantic similarity including word2vec, tfidf, lda, lsi.    
By changing setting in config.py:   
You can change the output vector dimension of each model except tfidf (dimension depends on corpus size).    
You can combine more than one models to generate semantic vectors of texts.

## Contact
As a beginner interested in NLP/Data Mining, I would be delighted for your encouragement!    
Feel free to mail me at 953383269@qq.com with any comments/problems/questions/suggestions.
