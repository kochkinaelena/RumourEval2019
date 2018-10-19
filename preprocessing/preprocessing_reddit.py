#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains functions for lading reddit data
"""

import os
import json
from copy import deepcopy
import numpy as np
from tree2branches import tree2branches
#%%
def listdir_nohidden(path):
    
    contents = os.listdir(path)
    new_contents = [i for i in contents if i[0] != '.']
    
    return new_contents
#%%
    
def load_data():
    
    # this is mix of twitter and reddit
    path_dev = "../rumoureval-2019-training-data/dev-key.json"
    with open(path_dev, 'r') as f:
        dev_key = json.load(f)
            
    path_train = "../rumoureval-2019-training-data/train-key.json"
    with open(path_train, 'r') as f:
        train_key = json.load(f)
    
    #%%
        
    path = "../rumoureval-2019-training-data/reddit-training-data"
    
    conversation_ids = listdir_nohidden(path)
    conversations = {}
    
    conversations['dev'] = []
    conversations['train'] = []
    conversations['test'] = []
   
    for id in conversation_ids:
        conversation = {}
        conversation['id'] = id
        path_src = path+'/'+id+'/source-tweet'
        files_t = sorted(listdir_nohidden(path_src))
        with open(os.path.join(path_src, files_t[0])) as f:
                for line in f:
                    src = json.loads(line)
                    
                    src['text'] = src['data']['children'][0]['data']['title']
                    src['user'] = src['data']['children'][0]['data']['author']
                    
                    if files_t[0].endswith('.json'):
                        filename = files_t[0][:-5]
                        src['id_str'] = filename
                    else:
                        print ("No, no I don't like that")
                    
                    src['used'] = 0
      
                    if src['id_str'] in list(dev_key['subtaskaenglish'].keys()):
                        src['setA'] = 'dev'
                        src['label'] = dev_key['subtaskaenglish'][src['id_str']]


                    elif src['id_str'] in list(train_key['subtaskaenglish'].keys()):
                        src['setA'] = 'train'
                        src['label'] = train_key['subtaskaenglish'][src['id_str']]

                    else:
                        
                        print ("Post was not found! Task A, Post ID: ", src['id_str'])
                                    
                    if src['id_str'] in list(dev_key['subtaskbenglish'].keys()):
                        src['setB'] = 'dev'
                        conversation['veracity'] = dev_key['subtaskbenglish'][src['id_str']]

                    elif src['id_str'] in list(train_key['subtaskbenglish'].keys()):
                        src['setB'] = 'train'
                        conversation['veracity'] = train_key['subtaskbenglish'][src['id_str']]

                    else:
                        print ("Post was not found! Task B, Post ID: ", src['id_str'])
                    
                    conversation['source'] = src
                    
                    
        tweets = []
        path_repl = path+'/'+id+'/replies'
        files_t = sorted(listdir_nohidden(path_repl))
        for repl_file in files_t:
            with open(os.path.join(path_repl, repl_file)) as f:
                for line in f:
                    tw = json.loads(line)

                    if 'body' in list(tw['data'].keys()):
                    
                        tw['text'] = tw['data']['body']
                        tw['user'] = tw['data']['author']
                        
                        if repl_file.endswith('.json'):
                            filename = repl_file[:-5]
                            tw['id_str'] = filename
                        else:
                            print ("No, no I don't like that reply")
                        
                        tw['used'] = 0
                        if tw['id_str'] in list(dev_key['subtaskaenglish'].keys()):
                            tw['setA'] = 'dev'
                            tw['label'] = dev_key['subtaskaenglish'][tw['id_str']]
                        elif tw['id_str'] in list(train_key['subtaskaenglish'].keys()):
                            tw['setA'] = 'train'
                            tw['label'] = train_key['subtaskaenglish'][tw['id_str']]
                        else:
                            print ("Post was not found! Task A, Reply ID: ", tw['id_str'])
                    
                        tweets.append(tw)
                    else:

                        tw['text'] = ''
                        tw['user'] = ''
                        tw['used'] = 0
                        if repl_file.endswith('.json'):
                            filename = repl_file[:-5]
                            tw['id_str'] = filename
                        else:
                            print ("No, no I don't like that reply")
                        if tw['id_str'] in list(dev_key['subtaskaenglish'].keys()):
                            tw['setA'] = 'dev'

                            tw['label'] = dev_key['subtaskaenglish'][tw['id_str']]
                        elif tw['id_str'] in list(train_key['subtaskaenglish'].keys()):
                            tw['setA'] = 'train'
                            tw['label'] = train_key['subtaskaenglish'][tw['id_str']]
                        else:
                            print ("Post was not found! Task A, Reply ID: ", tw['id_str'])
                        tweets.append(tw)
                        
        conversation['replies'] = tweets
        path_struct = path+'/'+id+'/structure.json'
      
        with open(path_struct, 'r') as f:
            struct = json.load(f)
            conversation['structure'] = struct
            branches = tree2branches(conversation['structure'])
            conversation['branches'] = branches

        conversations['train'].append(conversation)
#%%
    path = "/Users/Helen/Documents/PhD/SemEval2019/rumoureval-2019-training-data/reddit-dev-data"
    
    conversation_ids = listdir_nohidden(path)
   
    for id in conversation_ids:
        conversation = {}
        conversation['id'] = id
        path_src = path+'/'+id+'/source-tweet'
        files_t = sorted(listdir_nohidden(path_src))
        with open(os.path.join(path_src, files_t[0])) as f:
                for line in f:
                    src = json.loads(line)
                    
                    src['text'] = src['data']['children'][0]['data']['title']
                    src['user'] = src['data']['children'][0]['data']['author']
                    
                    if files_t[0].endswith('.json'):
                        filename = files_t[0][:-5]
                        src['id_str'] = filename
                    else:
                        print ("No, no I don't like that")
                    
                    src['used'] = 0
    #                
                    if src['id_str'] in list(dev_key['subtaskaenglish'].keys()):
                        src['setA'] = 'dev'
                        src['label'] = dev_key['subtaskaenglish'][src['id_str']]

                    elif src['id_str'] in list(train_key['subtaskaenglish'].keys()):
                        src['setA'] = 'train'

                        src['label'] = train_key['subtaskaenglish'][src['id_str']]

                    else:
                        print ("Post was not found! Task A, Post ID: ", src['id_str'])
                                    
                    if src['id_str'] in list(dev_key['subtaskbenglish'].keys()):
                        src['setB'] = 'dev'
                        conversation['veracity'] = dev_key['subtaskbenglish'][src['id_str']]

                    elif src['id_str'] in list(train_key['subtaskbenglish'].keys()):
                        src['setB'] = 'train'
                        conversation['veracity'] = train_key['subtaskbenglish'][src['id_str']]

                    else:
                        print ("Post was not found! Task B, Post ID: ", src['id_str'])
                    
                    conversation['source'] = src
                    
                    
        tweets = []
        path_repl = path+'/'+id+'/replies'
        files_t = sorted(listdir_nohidden(path_repl))
        for repl_file in files_t:
            with open(os.path.join(path_repl, repl_file)) as f:
                for line in f:
                    tw = json.loads(line)
                    if 'body' in list(tw['data'].keys()):
                    
                        tw['text'] = tw['data']['body']
                        tw['user'] = tw['data']['author']
                        
                        if repl_file.endswith('.json'):
                            filename = repl_file[:-5]
                            tw['id_str'] = filename
                        else:
                            print ("No, no I don't like that reply")
                        
                        tw['used'] = 0
                        if tw['id_str'] in list(dev_key['subtaskaenglish'].keys()):
                            tw['setA'] = 'dev'
                            tw['label'] = dev_key['subtaskaenglish'][tw['id_str']]

                        elif tw['id_str'] in list(train_key['subtaskaenglish'].keys()):
                            tw['setA'] = 'train'
                            tw['label'] = train_key['subtaskaenglish'][tw['id_str']]
                        else:
                            print ("Post was not found! Task A, Reply ID: ", tw['id_str'])
                    
                        tweets.append(tw)
                    else:
                        tw['text'] = ''
                        tw['user'] = ''
                        tw['used'] = 0
                        if repl_file.endswith('.json'):
                            filename = repl_file[:-5]
                            tw['id_str'] = filename
                        else:
                            print ("No, no I don't like that reply")
                        if tw['id_str'] in list(dev_key['subtaskaenglish'].keys()):
                            tw['setA'] = 'dev'
                            tw['label'] = dev_key['subtaskaenglish'][tw['id_str']]
                        elif tw['id_str'] in list(train_key['subtaskaenglish'].keys()):
                            tw['setA'] = 'train'
                            tw['label'] = train_key['subtaskaenglish'][tw['id_str']]

                        else:
                            print ("Post was not found! Task A, Reply ID: ", tw['id_str'])
                        tweets.append(tw)
        conversation['replies'] = tweets
        path_struct = path+'/'+id+'/structure.json'

        with open(path_struct, 'r') as f:
            struct = json.load(f)
            conversation['structure'] = struct
            branches = tree2branches(conversation['structure'])
            conversation['branches'] = branches
       

        conversations['dev'].append(conversation)
    return conversations