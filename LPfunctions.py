### import.py
import json
import pandas as pd
def textimport():
    data = []
    with open('data/pan20-authorship-verification-training-small.jsonl') as f:
        for l in f:
            data.append(json.loads(l.strip()))
    return pd.DataFrame.from_dict(data)


def textimport_light():
    data = []
    with open('data/pan20-authorship-verification-training-small.jsonl') as f:
        for l in f.readlines(100000000):
            data.append(json.loads(l.strip()))
    return pd.DataFrame.from_dict(data)
