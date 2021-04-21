### import.py
import json
import pandas as pd
text = 'data/pan20-authorship-verification-training-small.jsonl'
def textimport(text):
    data = []
    with open(text) as f:
        for l in f:
            data.append(json.loads(l.strip()))
    return pd.DataFrame.from_dict(data)
