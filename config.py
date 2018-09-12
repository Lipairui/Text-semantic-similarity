# Set model config:
''' 
format: 
        model name: setting(int)
        if setting == 0: use model with a default dimension
        if setting > 0 : use model with the dimension to be dim
        if setting == -1: not use model
 
'''

config = {
    'lda':10,
    'lsi':0,
    'word2vec':0,
    'tfidf':0 # 0/-1
}