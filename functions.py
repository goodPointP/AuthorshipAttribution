### import.py
import pandas as pd
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import string
import pickle
import re
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics.pairwise import cosine_distances

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

def remove_punc(corpus):
    texts = []
    key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
    for text in corpus:
        texts.append(str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key)))
    return texts

def remove_stop(corpus):
    texts = []
    stop_words = stopwords.words('english')
    stopwords_dict = set(stop_words)
    for text in corpus:
        texts.append(' '.join([word for word in text.split() if word.lower() not in stopwords_dict]))
    return texts

def remove_punc_stop(corpus):
    stop_words = stopwords.words('english')
    key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
    stopwords_dict = set(stop_words)
    texts = []
    for text in corpus:
        text = str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key))
        texts.append(' '.join([word for word in text.split() if word.lower() not in stopwords_dict]))    
    return texts

def remove_spaces(text):
    return text.replace(' ','')

def tokenizer(text):
    word_tokenizer = RegexpTokenizer(r"\w+")
    w = word_tokenizer.tokenize(text)
    s = sent_tokenize(text)
    return w,s

def normalize(data):
    normalized = StandardScaler()
    norm_data = normalized.fit_transform(data)
    return norm_data

def split_data(data, truth):
    X_train, X_test, y_train, y_test = train_test_split(
    data, truth, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocessing_complete(corpus):
    corpus = list(corpus)
    minus_punc = remove_punc(corpus)
    minus_stop = remove_stop(corpus)
    minus_both = remove_punc_stop(corpus)
    return [corpus, minus_punc, minus_stop, minus_both]

def dist(inp1, inp2):
    return abs(inp1 - inp2)
    
def cosine(inp1, inp2):
    return cosine_distances(inp1, inp2)


