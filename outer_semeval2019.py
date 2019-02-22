
from numpy.random import seed
seed(364)
from tensorflow import set_random_seed
set_random_seed(364)
from parameter_search import parameter_search
from objective_functions import objective_function_stance_branchLSTM_RumEv
from objective_functions import objective_function_veracity_branchLSTM_RumEv
from evaluation_functions import evaluation_function_stance_branchLSTM_RumEv
from evaluation_functions import evaluation_function_veracity_branchLSTM_RumEv
import json

from sklearn.metrics import f1_score, accuracy_score

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
    
    with open("output/answer.json", 'w') as f:
        json.dump(answer, f)
        
    return answer
#%%
print ('Rumour Stance classification')

ntrials = 100
task = 'stance'
paramsA, trialsA = parameter_search(ntrials, objective_function_stance_branchLSTM_RumEv, task)
#%%
best_trial_id = trialsA.best_trial["tid"]
best_trial_loss = trialsA.best_trial["result"]["loss"]
dev_result_id = trialsA.attachments["ATTACH::%d::ID" % best_trial_id]
dev_result_predictions =trialsA.attachments["ATTACH::%d::Predictions" % best_trial_id]
dev_result_label =trialsA.attachments["ATTACH::%d::Labels" % best_trial_id]

print(accuracy_score(dev_result_label,dev_result_predictions))
print(f1_score(dev_result_label,dev_result_predictions,average='macro'))

#%%
test_result_id, test_result_predictions = evaluation_function_stance_branchLSTM_RumEv(paramsA)
#%%
print ('Rumour Veracity classification') 
ntrials = 100
task = 'veracity'
paramsB, trialsB = parameter_search(ntrials, objective_function_veracity_branchLSTM_RumEv, task)
#%%
best_trial_idB = trialsB.best_trial["tid"]
best_trial_lossB = trialsB.best_trial["result"]["loss"]
dev_result_idB = trialsB.attachments["ATTACH::%d::ID" % best_trial_id]
dev_result_predictionsB = trialsB.attachments["ATTACH::%d::Predictions" % best_trial_id]
dev_result_labelB = trialsB.attachments["ATTACH::%d::Labels" % best_trial_id]
#confidenceB = [1.0 for i in range((len(dev_result_predictionsB)))]

print(accuracy_score(dev_result_labelB,dev_result_predictionsB))
print(f1_score(dev_result_labelB,dev_result_predictionsB,average='macro'))

#%%
test_result_idB, test_result_predictionsB, confidenceB  = evaluation_function_veracity_branchLSTM_RumEv(paramsB)

#confidenceB = [1.0 for i in range((len(test_result_predictionsB)))]

#%%
#convertsave_competitionformat(dev_result_id, dev_result_predictions, dev_result_idB, dev_result_predictionsB,confidenceB )

a = convertsave_competitionformat(test_result_id, test_result_predictions, test_result_idB, test_result_predictionsB,confidenceB )
