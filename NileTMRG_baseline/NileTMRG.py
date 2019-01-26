#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:00:31 2017

@author: Helen
"""

import os 
import numpy as np
import json
import gensim
import nltk
import re
from nltk.corpus import stopwords
from copy import deepcopy
import pickle
from nltk.tokenize import TweetTokenizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from preprocess import read_data
from preprocess import get_features
#%%
# Read tweets from each of the sets
all_data = read_data()

#%%
# Get BOW
BOW_sents, all_extra_feats, Y, ids = get_features(all_data, whichset="training")

cv = CountVectorizer()

cv_fit=cv.fit_transform(BOW_sents)

BOW_features = cv_fit.toarray()

# Append features
X = []
for i in range(len(BOW_features)):
    line = list(BOW_features[i]) + all_extra_feats[i]
    X.append(line)

# scale 
#scl = StandardScaler()
#X = scl.fit_transform(X)
# fit classifier

clf = LinearSVC(random_state=364)
clf.fit(X, Y) 

test_BOW_sents, test_all_extra_feats, Y_test, ids_test = get_features(all_data, whichset="development")
# preprocess testing set

test_cv_fit = cv.transform(test_BOW_sents)
test_BOW_features = test_cv_fit.toarray()

# Append features
X_test = []
for i in range(len(test_BOW_features)):
    line = list(test_BOW_features[i]) + test_all_extra_feats[i]
    X_test.append(line)

#X_test = scl.transform(X_test)
# get predicitons
Y_pred = clf.predict(X_test)

# get scores
#print "Accuracy:"
#print accuracy_score(Y_test, Y_pred)
#
#print "Macro F:"
#print f1_score(Y_test, Y_pred, average='macro')
#
#print confusion_matrix(Y_test, Y_pred)


#%%
submission_B = {}
for i, id in enumerate(ids_test):
    
    submission_B[id] = [Y_pred[i], 1]
#    
#import json
#with open('submissionB.json', 'w') as outfile:
#    json.dump(submission_B, outfile)
#

#def labell2strB(label):
#    
#    if label == 0:
#        return("true")
#    elif label == 1:
#        return("false")
#    elif label == 2:
#        return("unverified")
#    else:
#        print(label)
    
subtaskaenglish = {}
subtaskbenglish = {}

#for i,id in enumerate(idsA):
#    subtaskaenglish[id] = labell2strA(predictionsA[i])

#for i,id in enumerate(idsB):
#    subtaskbenglish[id] = [labell2strB(predictionsB[i]),confidenceB[i]]

answer = {}
answer['subtaskaenglish'] = {}
answer['subtaskbenglish'] = submission_B

answer['subtaskadanish'] = {}
answer['subtaskbdanish'] = {}

answer['subtaskarussian'] = {}
answer['subtaskbrussian'] = {}

with open("answerB.json", 'w') as f:
    json.dump(answer, f)
        













































