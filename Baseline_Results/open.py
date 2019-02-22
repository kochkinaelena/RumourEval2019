"""
This code shows how to load files from this folder

"""

import json
import pickle
from sklearn.metrics import f1_score, accuracy_score
from keras.models import load_model
from keras.models import model_from_json
#%%
file = "answer.json" # this file contains output of branchLSTM model for Tasks A and B

with open(file, 'r') as f:
    resultLSTM = json.load(f)
    
#%%
    
file = "answerB_myNileTMRG.json" # this file contains output of my implementation of NileTMRG model for Task B

with open(file, 'r') as f:
    resultNile = json.load(f)
    
#%%

# this is to load hyperparameters used for branchLSTM model for task A
file = "bestparams_stance.txt"
with open(file, 'rb') as f:
    bestparams_stance = pickle.load(f)

#%%
    
# this is to load hyperparameters used for branchLSTM model for task B
    
file = "bestparams_veracity.txt"
with open(file, 'rb') as f:
    bestparams_veracity = pickle.load(f)


#%%

#this is to load and access saved data from each of the hyper parameter trial for task A (trained on train set and evaluated on developemtn)
    
file = "trials_stance.txt"
with open(file, 'rb') as f:
    trials_stance = pickle.load(f)
    
    
Abest_trial_id = trials_stance.best_trial["tid"]
Abest_trial_loss = trials_stance.best_trial["result"]["loss"]
Adev_result_id = trials_stance.attachments["ATTACH::%d::ID" % Abest_trial_id]
Adev_result_predictions = trials_stance.attachments["ATTACH::%d::Predictions" % Abest_trial_id]
Adev_result_label = trials_stance.attachments["ATTACH::%d::Labels" % Abest_trial_id]

#%%

#this is to load and access saved data from each of the hyper parameter trial for task B (trained on train set and evaluated on developemtn)

file = "trials_veracity.txt"
with open(file, 'rb') as f:
    trials_veracity = pickle.load(f)
    
Bbest_trial_id = trials_veracity.best_trial["tid"]
Bbest_trial_loss = trials_veracity.best_trial["result"]["loss"]
Bdev_result_id = trials_veracity.attachments["ATTACH::%d::ID" % Bbest_trial_id]
Bdev_result_predictions = trials_veracity.attachments["ATTACH::%d::Predictions" % Bbest_trial_id]
Bdev_result_label = trials_veracity.attachments["ATTACH::%d::Labels" % Bbest_trial_id]

#%%
    
# this is two ways to load a model that was used to obtain final result for Task A (trained on train+dev and evaluated on test data)

modelA = load_model('my_model_stance.h5')

# OR

file = "model_architecture_stance.json"
with open(file, 'r') as f:
    json_string = json.load(f)

modelA = model_from_json(json_string)
modelA.load_weights('my_model_stance_weights.h5')

#%%
    
# this is two ways to load a model that was used to obtain final result for Task B (trained on train+dev and evaluated on test data)
modelB = load_model('my_model_veracity.h5')

# OR

file = "model_architecture_veracity.json"
with open(file, 'r') as f:
    json_string = json.load(f)

modelB = model_from_json(json_string)
modelB.load_weights('my_model_veracity_weights.h5')


