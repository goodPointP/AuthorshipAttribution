### POS-tagging.py #0s
import pandas as pd
import numpy as np
from functions import textimport_light, truthimport_light, read_data, read_truth_data
from collections import Counter, OrderedDict, ChainMap
import spacy
import string
import re
from sklearn import svm
import operator, functools
from nltk.util import skipgrams
from nltk.corpus import stopwords

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


#%% defining the subset used (no. of texts) 0.06s
print("3rd part")

key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
no = int(50)
texts = ['"{}"'.format(str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key))) for text in text_uniques[:no]]

# 1. change quotes to apostrophes to make sure contractions like "aren't" are detected as such (JSON changes apostrohes to quotes when loading) == text.replace('"', "'"))
# 2. remove apostrophes that ARE NOT between letters (essentially just remove quotes) == re.sub("\\s*'\\B|\\B'\\s*" 
# 3. remove punctuation (key = punctuation except apostrophe and quote) == .translate(key))
# 4. wrap each string in start/end quotes. == '"{}"'.format

#%% Removing stopwords

# stopwordkey = str.maketrans('','', str(stopwords.words('english')))
# test2 = str(test.split()).translate(stopwordkey)

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
#%%
print("7th part")

tagged_pairs = np.array(vecs, dtype="object")[np.array(list(df['text_id'][:int(no/2)]))]
x, y, z = tagged_pairs.shape
tagged_pairs = tagged_pairs.reshape(x, y*z)


#%% SVM
print("8th part")

X = tagged_pairs
y = np.array(df_truth['same'][:int(no/2)]).astype(int)
svm_clf = svm.SVC()
svm_clf.fit(X, y)
svm_clf.predict(X[0].reshape(1,-1))