#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outer file for semeval 2019 baseline

"""
from parameter_search import parameter_search
from objective_functions import objective_function_stance_branchLSTM_RumEv
from objective_functions import objective_function_veracity_branchLSTM_RumEv
import json
#%%
def labell2strA(label):
    
    if label == 0:
        return("support")
    elif label == 1:
        return("comment")
    elif label == 2:
        return("deny")
    elif label == 3:
        return("query")
    else:
        print(label)
    

def labell2strB(label):
    
    if label == 0:
        return("true")
    elif label == 1:
        return("false")
    elif label == 2:
        return("unverified")
    else:
        print(label)

#%%

def convertsave_competitionformat(idsA, predictionsA, idsB, predictionsB, confidenceB):
    
    subtaskaenglish = {}
    subtaskbenglish = {}
    
    for i,id in enumerate(idsA):
        subtaskaenglish[id] = labell2strA(predictionsA[i])
    
    for i,id in enumerate(idsB):
        subtaskbenglish[id] = [labell2strB(predictionsB[i]),confidenceB[i]]
    
    answer = {}
    answer['subtaskaenglish'] = subtaskaenglish
    answer['subtaskbenglish'] = subtaskbenglish
    
    answer['subtaskadanish'] = {}
    answer['subtaskbdanish'] = {}
    
    answer['subtaskarussian'] = {}
    answer['subtaskbrussian'] = {}
    
    with open("answer.json", 'w') as f:
        json.dump(answer, f)
#%%
print ('Rumour Stance classification')

ntrials = 50
task = 'stance'
params, trials = parameter_search(ntrials, objective_function_stance_branchLSTM_RumEv, task)
#%%
best_trial_id = trials.best_trial["tid"]
best_trial_loss = trials.best_trial["result"]["loss"]
dev_result_id = trials.attachments["ATTACH::%d::ID" % best_trial_id]
dev_result_predictions =trials.attachments["ATTACH::%d::Predictions" % best_trial_id]
dev_result_label =trials.attachments["ATTACH::%d::Labels" % best_trial_id]

#%%
print ('Rumour Veracity classification') 
ntrials = 50
task = 'veracity'
params, trials = parameter_search(ntrials, objective_function_veracity_branchLSTM_RumEv, task)
#%%
best_trial_idB = trials.best_trial["tid"]
best_trial_lossB = trials.best_trial["result"]["loss"]
dev_result_idB = trials.attachments["ATTACH::%d::ID" % best_trial_id]
dev_result_predictionsB = trials.attachments["ATTACH::%d::Predictions" % best_trial_id]
dev_result_labelB = trials.attachments["ATTACH::%d::Labels" % best_trial_id]
confidenceB = [1.0 for i in range((len(dev_result_predictionsB)))]
#%%
convertsave_competitionformat(dev_result_id, dev_result_predictions, dev_result_idB, dev_result_predictionsB,confidenceB )