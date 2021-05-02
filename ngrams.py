from functions import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import time

#%%

raw_data = read_data()
dataframe = textimport_light(raw_data, pandas_check = True)
text_uniques = remove_duplicates(dataframe)[1]


#%%

def count_word_ngrams(dataset, min_n, max_n):
    c_w_ngrams = CountVectorizer(ngram_range = (min_n, max_n))
    c_w_ngrams_vec = c_w_ngrams.fit_transform(dataset).toarray()
 
    return c_w_ngrams_vec

def count_char_ngrams(dataset, min_n, max_n):
    c_c_ngrams = CountVectorizer(analyzer = "char", ngram_range = (min_n, max_n))
    c_c_ngrams_vec = c_c_ngrams.fit_transform(dataset).toarray()

    return c_c_ngrams_vec

def tfidf_word_ngrams(dataset, min_n, max_n):
    t_w_ngrams = TfidfVectorizer(ngram_range = (min_n, max_n))
    t_w_ngrams_vec = t_w_ngrams.fit_transform(dataset).toarray()

    return t_w_ngrams_vec

def tfidf_char_ngrams(dataset, min_n, max_n):
    t_c_ngrams = TfidfVectorizer(analyzer = "char", ngram_range = (min_n, max_n))
    t_c_ngrams_vec = t_c_ngrams.fit_transform(dataset).toarray()
    
    return t_c_ngrams_vec

#the code I removed == c_w_ngrams_vec[np.where(dataset == text)]

start = time.time()
one = count_word_ngrams(text_uniques[0:100], 1, 3)
#two = count_char_ngrams(text_uniques[0], text_uniques, 3)
#three = tfidf_word_ngrams(text_uniques[0], text_uniques, 3)
#four = tfidf_char_ngrams(text_uniques[0], text_uniques, 3)
end = time.time()

print(end - start)