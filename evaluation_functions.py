#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code contains several versions of evaluation functions
"""
import numpy as np
from LSTM_models import LSTM_model_stance,LSTM_model_veracity
#from LSTM_models import build_LSTM_model_veracity
#from rmse import rmse
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels_test
import pickle
from copy import deepcopy
#%%


def evaluation_function_stance_branchLSTM_RumEv(params):
    
    path = "preprocessing/saved_dataRumEval2019"
    x_train = np.load(os.path.join(path, 'train/train_array.npy'))
    y_train = np.load(os.path.join(path, 'train/fold_stance_labels.npy'))

    print (x_train.shape)

#    ids_train = np.load(os.path.join(path, 'train/tweet_ids.npy'))
    x_dev = np.load(os.path.join(path, 'dev/train_array.npy'))
    y_dev = np.load(os.path.join(path, 'dev/fold_stance_labels.npy'))
#    ids_dev = np.load(os.path.join(path, 'dev/tweet_ids.npy'))
    x_test = np.load(os.path.join(path, 'test/train_array.npy'))
#    y_test = np.load(os.path.join(path, 'test/fold_stance_labels.npy'))
    ids_test = np.load(os.path.join(path, 'test/tweet_ids.npy'))
    # join dev and train
    x_dev = pad_sequences(x_dev, maxlen=len(x_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    y_dev = pad_sequences(y_dev, maxlen=len(y_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    
    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_train = np.concatenate((y_train, y_dev), axis=0)
    y_train_cat = []
    for i in range(len(y_train)):
        y_train_cat.append(to_categorical(y_train[i], num_classes=4))
    y_train_cat = np.asarray(y_train_cat)
    y_pred, _ = LSTM_model_stance(x_train, y_train_cat,
                                           x_test, params,eval=True )
    # get tree labels
#    pad ids first
#%%
    fids_test = []
    
    width = y_pred.shape[1]
    
    for i in ids_test:
        for j in range(width):
            if j<len(i):
                fids_test.append(i[j])
            else:
                fids_test.append('0')
        
    
#%%    
    
#    fids_test = ids_test.flatten()
    fy_pred = y_pred.flatten()
#    fy_test = y_test.flatten()
#    fconfidence = confidence.flatten()
    uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
    uniqtwid = uniqtwid.tolist()
    uindices2 = uindices2.tolist()
    if uniqtwid[0]=='0':
        del uniqtwid[0]
        del uindices2[0]
    else:
        delind = uniqtwid.index('0')
        del uniqtwid[delind]
        del uindices2[delind]
        
    uniq_dev_prediction = [fy_pred[i] for i in uindices2]
#    uniq_dev_label = [fy_test[i] for i in uindices2]
#    uniq_dev_confidence = [fconfidence[i] for i in uindices2]
    
    return uniqtwid, uniq_dev_prediction

#%%


def evaluation_function_veracity_branchLSTM_RumEv(params):

    path = 'preprocessing/saved_dataRumEval2019'
    x_train = np.load(os.path.join(path, 'train/train_array.npy'))
    y_train = np.load(os.path.join(path, 'train/labels.npy'))
    x_dev = np.load(os.path.join(path, 'dev/train_array.npy'))
    y_dev = np.load(os.path.join(path, 'dev/labels.npy'))
    x_test = np.load(os.path.join(path, 'test/train_array.npy'))
    ids_test = np.load(os.path.join(path, 'test/ids.npy'))
    # join dev and train
    x_dev = pad_sequences(x_dev, maxlen=len(x_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_train = np.concatenate((y_train, y_dev), axis=0)
    y_train = to_categorical(y_train, num_classes=None)
    y_pred, confidence = LSTM_model_veracity(x_train, y_train, x_test, params, eval=True)
    # get tree labels
    
    trees, tree_prediction, tree_confidence = branch2treelabels_test(
                                                                ids_test,
                                                                y_pred,
                                                                confidence)
    
        
    return trees, tree_prediction, tree_confidence
