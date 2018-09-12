# text-semantic-similarity
Calculate semantic similarity of two texts.  
Models include word2vec, tfidf, lda, lsi.
## Dependencies
Python 3.6.5
pandas, numpy, jieba, nltk, gensim, re, sklearn, codecs, time
## Implementation
Since semantic similarity labeling is difficult and time consuming, unsupervised semantic similarity caculating is useful.
I choose four models to calculate semantic vectors of texts, then utilize cosine distance to calculate similarity.
## How to use?
command: excute.py text1_path text2_path res_path -l en/cn
You can run excute.py with -h to get information about arguments details.
## How to change model setting?
I implement four models to calculate unsupervised semantic similarity including word2vec, tfidf, lda, lsi.
You can change the output vector dimension of each model except tfidf (dimension depends on corpus size).
You can combine more than one models to generate semantic vectors of texts.
