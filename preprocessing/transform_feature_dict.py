"""
This code helps convert dictionaries of features from conversation into
arrays of branches of conversation
"""

import numpy as np
from tree2branches import tree2branches

#%%
def convert_label(label):
    if label == "support":
        return(0)
    elif label == "comment":
        return(1)
    elif label == "deny":
        return(2)
    elif label == "query":
        return(3)
    else:
        print(label)
#%%
def transform_feature_dict(thread_feature_dict, conversation, feature_set):
    thread_features_array = []
    thread_stance_labels = []
    clean_branches = []

    branches = conversation['branches']

    for branch in branches:
        branch_rep = []
        clb = []
        branch_stance_lab = []
        for twid in branch:
            if twid in thread_feature_dict.keys():
                tweet_rep = dict_to_array(thread_feature_dict[twid],
                                          feature_set)
                branch_rep.append(tweet_rep)

                if twid == branch[0]:
                    if 'label' in list(conversation['source'].keys()):
                        branch_stance_lab.append(convert_label(
                                            conversation['source']['label']))
                    clb.append(twid)
                else:
                    for r in conversation['replies']:
                        if r['id_str'] == twid:
                            if 'label' in list(r.keys()):

                                branch_stance_lab.append(
                                                    convert_label(r['label']))
                            clb.append(twid)
        if branch_rep != []:
            branch_rep = np.asarray(branch_rep)
            branch_stance_lab = np.asarray(branch_stance_lab)
            thread_features_array.append(branch_rep)
            thread_stance_labels.append(branch_stance_lab)
            clean_branches.append(clb)
     
    return thread_features_array, thread_stance_labels, clean_branches

#%%
def dict_to_array(feature_dict, feature_set):

    tweet_rep = []
    for feature_name in feature_set:

        if np.isscalar(feature_dict[feature_name]):
            tweet_rep.append(feature_dict[feature_name])
        else:
            tweet_rep.extend(feature_dict[feature_name])
    tweet_rep = np.asarray(tweet_rep)
    return tweet_rep
