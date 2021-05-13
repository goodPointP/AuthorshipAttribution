### POS-tagging.py #0s
import pandas as pd
import numpy as np
from functions import textimport_light, truthimport_light, read_data, read_truth_data
from collections import Counter,  ChainMap
import spacy
import string
import re
from nltk.util import skipgrams
from scipy import spatial
import itertools

#%%
def importDataPOS():
    rawData = read_data()
    rawTruths = read_truth_data()
    df = textimport_light(rawData)
    text_list = pd.Series(df['pair'].explode())
    text_IDs, text_uniques = text_list.factorize()
    df['text_id'] = pd.Series(zip(text_IDs[0::2], text_IDs[1::2]))
    return df

#%%

def pos_tag(ntexts):
    
    
    df = importDataPOS()
    text_list = pd.Series(df['pair'].explode())[:ntexts]
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "senter","ner"])
    
    #preprocessing
    text_IDs, text_uniques = text_list.factorize()
    key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
    texts = ['"{}"'.format(str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key))) for text in text_uniques]

    #pos-tagging
    tags = []
    for doc in nlp.pipe(texts, batch_size=5):
        tags.append([token.tag_ for token in doc])
        
    return tags
    
def skipgramming(tags, return_counts=False):
    #skipgram-creation
    df = importDataPOS()
    
    no = int(len(tags))
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
    
    #cosine similarity
    tagged_pairs = tf_idf[np.array(list(df['text_id'][:int(no/2)]))]
    cos = [spatial.distance.pdist(pair, metric='cosine') for pair in tagged_pairs]
    
    if return_counts==True:
        pos_counts = [dict(Counter(text)) for text in tags]
        return cos, pos_counts
    else:
        return np.stack(cos)