"""
This file contains utility functions for preprocessing
"""
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from copy import deepcopy
import gensim
#%%


def str_to_wordlist(tweettext, tweet, remove_stopwords=False):
    str_text = re.sub("[^a-zA-Z]", " ", tweettext)
    words = nltk.word_tokenize(str_text.lower())
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return(words)

def loadW2vModel():
    # LOAD PRETRAINED MODEL
    global model_GN
    print ("Loading the model")
    model_GN = gensim.models.KeyedVectors.load_word2vec_format(
                    '/Users/Helen/Documents/PhD/Pre-trained WORD2VEC/GoogleNews-vectors-negative300.bin', binary=True)
    print ("Done!")

def sumw2v(tweet, avg=True):
    global model_GN
    model = model_GN
    num_features = 300
    temp_rep = np.zeros(num_features)
    wordlist = str_to_wordlist(tweet['text'], tweet, remove_stopwords=False)
    for w in range(len(wordlist)):
        if wordlist[w] in model:
            temp_rep += model[wordlist[w]]
    if avg and len(wordlist) != 0:
        sumw2v = temp_rep/len(wordlist)
    else:
        sumw2v = temp_rep
    return sumw2v

def getW2vCosineSimilarity(words, wordssrc):
    global model_GN
    model = model_GN
    words2 = []
    for word in words:
        if word in model.wv.vocab:  # change to model.wv.vocab
            words2.append(word)
    wordssrc2 = []
    for word in wordssrc:
        if word in model.wv.vocab:  # change to model.wv.vocab
            wordssrc2.append(word)
    if len(words2) > 0 and len(wordssrc2) > 0:
        return model.n_similarity(words2, wordssrc2)
    return 0.