### import.py
import pandas as pd
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import string
import pickle

def read_data():
    with open('data/dataShuffled.pkl', 'rb') as f:
        rawData = pickle.load(f)
    return rawData

def read_truth_data():
    with open('data/labelsShuffled.pkl', 'rb') as f:
        rawTruths = pickle.load(f)
    return rawTruths

def textimport(rawData, pandas_check = True):
    if (pandas_check):
        return pd.DataFrame.from_dict(rawData)
    else:
        return list(rawData)

def truthimport(rawTruths, pandas_check = True):
    if (pandas_check):
        return pd.DataFrame.from_dict(rawTruths)
    else:
        return list(rawTruths)

def textimport_light(rawData, pandas_check = True):
    if (pandas_check):
        return pd.DataFrame.from_dict(rawData)[:2289]
    else:
        return list(rawData)[:2289]

def truthimport_light(rawTruths, pandas_check = True):
    if (pandas_check):
        return pd.DataFrame.from_dict(rawTruths)[:2289]
    else:
        return list(rawTruths)[:2289]

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
    

