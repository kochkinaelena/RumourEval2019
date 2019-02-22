#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get the performance numbers

"""
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

#%%
submission_file = "answer.json" # this file contains output of branchLSTM model for Tasks A and B
submission_fileB = "answerB_myNileTMRG.json"
reference_file = "final-eval-key.json"


with open(submission_file, 'r') as f:
    resultLSTM = json.load(f)
    
with open(submission_fileB, 'r') as f:
    resultNile = json.load(f)
    
with open(reference_file, 'r') as f:
    labels = json.load(f)


#%%
def convert_stance(labelint):
    convertdict = {}
    convertdict[0] = 'support'
    convertdict[1] = 'comment'
    convertdict[2] = 'deny'
    convertdict[3] = 'query'
    return convertdict[labelint]

def convert_veracity(labelint):
    convertdict = {}
    convertdict[0] = 'true'
    convertdict[1] = 'false'
    convertdict[2] = 'unverified'

    return convertdict[labelint]

#%%

task  = "subtaskaenglish"    
task_results = resultLSTM[task]
task_labels  = labels[task]

true = []
pred = []
for k in task_labels.keys():
    
    true.append(task_labels[k])
    pred.append(task_results[k])

print (task)
print ("Macro F score ", f1_score(true,pred, average = 'macro'))
print ("Accuracy ",accuracy_score(true,pred))
print ("Confusion matrix \n",confusion_matrix(true,pred, labels=['support','comment','deny','query']))
print ("Precision, recall, fscore, support per class \n",precision_recall_fscore_support(true,pred, labels=['support','comment','deny','query']))


#%%

task  = "subtaskbenglish"    
task_results = resultLSTM[task]
task_labels  = labels[task]

true = []
pred = []
for k in task_labels.keys():
    
    true.append(task_labels[k])
#    pred.append(task_results[k][0]) # if we simply do that, we won't get the same performance as in competition scorer
    
    yhat, confidence = task_results[k]
    if confidence>=0.5:
        pred.append(yhat)
    else:
        pred.append('unverified')
    

print (task)
print ("Macro F score ", f1_score(true,pred, average = 'macro'))
print ("Confusion matrix \n",confusion_matrix(true,pred, labels=['true','false','unverified']))
print ("Precision, recall, fscore, support per class \n",precision_recall_fscore_support(true,pred, labels=['true','false','unverified']))

#%%

task  = "subtaskbenglish"    
task_results = resultNile[task]
task_labels  = labels[task]

true = []
pred = []
for k in task_labels.keys():
    
    true.append(task_labels[k])
#    pred.append(task_results[k][0])
    
    yhat, confidence = task_results[k]
    if confidence>=0.5:
        pred.append(yhat)
    else:
        pred.append('unverified')

print (task)
print ("Macro F score ", f1_score(true,pred, average = 'macro'))
print ("Confusion matrix \n",confusion_matrix(true,pred, labels=['true','false','unverified']))
print ("Precision, recall, fscore, support per class \n",precision_recall_fscore_support(true,pred, labels=['true','false','unverified']))




