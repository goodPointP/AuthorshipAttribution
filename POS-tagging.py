### POS-tagging.py #0s
import pandas as pd
import numpy as np
from functions import textimport_light, truthimport_light, read_data, read_truth_data
from collections import Counter,  ChainMap
import spacy
import string
import re
from sklearn import svm
from nltk.util import skipgrams
import functools
import operator

#%% Read the data

rawData = read_data()
rawTruths = read_truth_data()

#%% Creating our corpus of texts. Labels and IDs are not included. #11s
print("first part")

df = textimport_light(rawData)
df_truth = truthimport_light(rawTruths)


#%% For splitting the textpairs into an array of 1xN. #0.3s
print("2nd part")

text_list = pd.Series(df['pair'].explode())
text_IDs, text_uniques = text_list.factorize()
df['text_id'] = pd.Series(zip(text_IDs[0::2], text_IDs[1::2]))

#%%
nlp = spacy.load("en_core_web_sm", exclude=["parser", "senter","ner"])

def pos_tag(text_list, no=100):
    
    # text_list = pd.Seties(df['pair'].explode()) or equal
    # no = number of texts used
    
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "senter","ner"])
    no = int(no)
    
    #preprocessing
    text_IDs, text_uniques = text_list.factorize()
    key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
    texts = ['"{}"'.format(str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key))) for text in text_uniques[:no]]

    #pos-tagging
    tags = []
    for doc in nlp.pipe(texts, batch_size=50):
        tags.append([token.tag_ for token in doc])
    
    #skipgram-creation
    skips = [dict(Counter(skipgrams(tagset, 2, 2)).most_common(200)) for tagset in tags]
    base = dict.fromkeys(dict(ChainMap(*skips)), 0)

    #ordering
    vecs = []
    for subset in skips:
        sub_dict = base.copy()
        for key, value in subset.items():
            sub_dict[key] = value
        vecs.append(list(sub_dict.values()))
    
    #TF
    doc_lengths = np.array([len(tagset) for tagset in tags]).reshape(-1,1)
    vec_array = np.array(vecs)
    term_freq = np.divide(vec_array, doc_lengths)

    #IDF
    vecs_transposed = np.array(vecs).T
    occurences = np.nonzero(vecs_transposed)[0]
    occurence_counts = np.unique(occurences, return_counts=True)[1]
    doc_freq = np.log(no/occurence_counts)

    #TF-IDF
    tf_idf = term_freq*doc_freq
    
    return tf_idf
#%% defining the subset used (no. of texts) 0.06s
print("3rd part")

key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
no = int(100)
texts = ['"{}"'.format(str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key))) for text in text_uniques[:no]]

# 1. change quotes to apostrophes to make sure contractions like "aren't" are detected as such (JSON changes apostrohes to quotes when loading) == text.replace('"', "'"))
# 2. remove apostrophes that ARE NOT between letters (essentially just remove quotes) == re.sub("\\s*'\\B|\\B'\\s*" 
# 3. remove punctuation (key = punctuation except apostrophe and quote) == .translate(key))
# 4. wrap each string in start/end quotes. == '"{}"'.format


#%% initializing spacy and excluding redundancy. #extracting POS-tags from texts. 11.6s
print("4th part")

nlp = spacy.load("en_core_web_sm", exclude=["parser", "senter","ner"])

tags = []
for doc in nlp.pipe(texts, batch_size=50):
    tags.append([token.tag_ for token in doc])
    
#%% recombining the pairs of texts as pairs of POS-tags. #Counting POS-tags. 0.1s
print("5th part")

skips = [dict(Counter(skipgrams(tagset, 2, 2)).most_common(200)) for tagset in tags]
base = dict.fromkeys(dict(ChainMap(*skips)), 0)

#%%
print("6th part")

vecs = []
for subset in skips:
    sub_dict = base.copy()
    for key, value in subset.items():
        sub_dict[key] = value
    vecs.append(list(sub_dict.values()))

#%% TF-IDF

#TF
doc_lengths = np.array([len(tagset) for tagset in tags]).reshape(-1,1)
vec_array = np.array(vecs)
term_freq = np.divide(vec_array, doc_lengths)

#IDF
vecs_transposed = np.array(vecs).T
occurences = np.nonzero(vecs_transposed)[0]
occurence_counts = np.unique(occurences, return_counts=True)[1]
doc_freq = np.log(no/occurence_counts)

#TF-IDF
tf_idf = term_freq*doc_freq

#%%
print("7th part")

tagged_pairs = tf_idf[np.array(list(df['text_id'][:int(no/2)]))]
x_d, y_d, z_d = tagged_pairs.shape
tagged_pairs = tagged_pairs.reshape(x_d, y_d*z_d)


#%% SVM
print("8th part")

X = tagged_pairs
X_train = X[:35]
X_test = X[35:]

y = np.array(df_truth['same'][:int(no/2)]).astype(int)
y_train = y[:35]
y_test = y[35:]
#%%

svm_clf = svm.SVC(kernel='rbf', C=3, gamma=100)
svm_clf.fit(X_train, y_train)
score = svm_clf.score(X_test, y_test)
