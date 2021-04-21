# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:53:06 2021

@author: Group
"""

import json
import pandas as pd

data = []
with open('data/pan20-authorship-verification-training-small.jsonl') as f:
    for l in f:
        data.append(json.loads(l.strip()))
        
df = pd.DataFrame.from_dict(data)

#%%

df = df.head(1000)
