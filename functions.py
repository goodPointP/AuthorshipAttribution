### import.py
import json
import pandas as pd
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import string

def textimport():
    data = []
    with open('data/pan20-authorship-verification-training-small.jsonl') as f:
        for l in f:
            data.append(json.loads(l.strip()))
    return pd.DataFrame.from_dict(data) 

def truthimport():
    data_truth = []
    with open('data/pan20-authorship-verification-training-small-truth.jsonl') as f:
        for l in f:
            data_truth.append(json.loads(l.strip()))
    return pd.DataFrame.from_dict(data_truth)

def textimport_light():
    data = []
    with open('data/pan20-authorship-verification-training-small.jsonl') as f:
        for l in f.readlines(100000000):
            data.append(json.loads(l.strip()))
    return pd.DataFrame.from_dict(data)

def truthimport_light():
    data_truth = []
    with open('data/pan20-authorship-verification-training-small-truth.jsonl') as f:
        for l in f:
            data_truth.append(json.loads(l.strip()))
    return pd.DataFrame.from_dict(data_truth)[:2289]

def remove_duplicates(dataframe):
    text_list = pd.Series(dataframe['pair'].explode())
    text_IDs, text_uniques = text_list.factorize()
    return text_IDs, text_uniques

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_spaces(text):
    return text.replace(' ','')

def tokenizer(text):
    word_tokenizer = RegexpTokenizer(r"\w+")
    w = word_tokenizer.tokenize(text)
    s = sent_tokenize(text)
    return w,s

def normalize(data):
    normalized = StandardScaler()
    normalized.fit(data)
    return normalized

def split_data(data, truth):
    X_train, X_test, y_train, y_test = train_test_split(
    data, truth, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test
    

