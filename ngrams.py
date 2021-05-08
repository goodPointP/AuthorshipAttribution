from functions import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time

#%%

raw_data = read_data()
df = textimport_light(raw_data, pandas_check = True)
text_IDs, text_uniques = remove_duplicates(df)
df['text_id'] = pd.Series(zip(text_IDs[0::2], text_IDs[1::2]))

#%%

def tfidf_word_ngrams(corpus, min_n, max_n):
    t_w_ngrams = TfidfVectorizer(ngram_range = (min_n, max_n), max_features=200)
    t_w_ngrams_vec = t_w_ngrams.fit_transform(corpus).toarray()
    
    no = int(len(corpus))
    tagged_pairs = t_w_ngrams_vec[np.array(list(df['text_id'][:int(no/2)]))]
    cos = [spatial.distance.pdist(pair, metric='cosine') for pair in tagged_pairs]

    return np.concatenate(cos, axis = 0)

def tfidf_char_ngrams(corpus, min_n, max_n):
    t_c_ngrams = TfidfVectorizer(analyzer = "char", ngram_range = (min_n, max_n), max_features=200)
    t_c_ngrams_vec = t_c_ngrams.fit_transform(corpus).toarray()
    
    no = int(len(corpus))
    tagged_pairs = t_c_ngrams_vec[np.array(list(df['text_id'][:int(no/2)]))]
    cos = [spatial.distance.pdist(pair, metric='cosine') for pair in tagged_pairs]

    return np.concatenate(cos, axis = 0)


start = time.time()
#three = tfidf_word_ngrams(text_uniques, 1, 1) # 111s
four = tfidf_char_ngrams(text_uniques[0:100], 3, 3)
end = time.time()

print(end - start)

#%%
