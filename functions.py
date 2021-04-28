### import.py
import pandas as pd
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import string
import pickle

def read_data():
    with open('data/dataShuffled.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def read_truth_data():
    with open('data/dataShuffled.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def textimport(pandas_check = True):
    if 'data' not in globals():
        data = read_data()
    if (pandas_check):
        return pd.DataFrame.from_dict(data)
    else:
        return list(data)

def truthimport(pandas_check = True):
    if 'data_truth' not in globals():
        data_truth = read_data()
    if (pandas_check):
        return pd.DataFrame.from_dict(data_truth)
    else:
        return list(data_truth)

def textimport_light(pandas_check = True):
    if 'data' not in globals():
        data = read_data()
    if (pandas_check):
        return pd.DataFrame.from_dict(data)
    else:
        return list(data)[:2289]

def truthimport_light(pandas_check = True):
    if 'data_truth' not in globals():
        data_truth = read_data()
    if (pandas_check):
        return pd.DataFrame.from_dict(data_truth)
    else:
        return list(data_truth)[:2289]

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
    

