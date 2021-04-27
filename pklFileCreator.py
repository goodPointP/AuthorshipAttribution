# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:34:23 2021

@author: John
"""
from functions import *
import random


data = textimport(False)
labels = truthimport(False)

c = list(zip(data, labels))

random.Random(1234).shuffle(c)

data, labels = zip(*c)

#%%
import pickle
with open('labelsShuffled.pkl', 'wb') as outfile:
    pickle.dump(labels, outfile)
    

#%%
labelPKLTEST=[]
with open('labelsShuffled.pkl', 'rb') as f:
    labelsPKLTEST = pickle.load(f)
#%%
with open('dataShuffled.pkl') as f:
        for l in f:
            dataPKLTEST.append(json.loads(l.strip()))