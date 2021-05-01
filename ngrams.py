from functions import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import time

#%%

raw_data = read_data()
dataframe = textimport_light(raw_data, pandas_check = True)
text_uniques = remove_duplicates(dataframe)[1]


#%%

def count_word_ngrams(text, dataset, n):
    c_w_ngrams = CountVectorizer(ngram_range = (n,n))
    c_w_ngrams_vec = c_w_ngrams.fit_transform(dataset).toarray()
    text_vec = c_w_ngrams_vec[np.where(dataset == text)]
    
    return text_vec.T, c_w_ngrams_vec

def count_char_ngrams(text, dataset, n):
    c_c_ngrams = CountVectorizer(analyzer = "char", ngram_range = (n,n))
    c_c_ngrams_vec = c_c_ngrams.fit_transform(dataset).toarray()
    text_vec = c_c_ngrams_vec[np.where(dataset == text)]
    
    return text_vec.T, c_c_ngrams_vec

def tfidf_word_ngrams(text, dataset, n):
    t_w_ngrams = TfidfVectorizer(ngram_range = (n,n))
    t_w_ngrams_vec = t_w_ngrams.fit_transform(dataset).toarray()
    text_vec = t_w_ngrams_vec[np.where(dataset == text)]
    
    return text_vec.T, t_w_ngrams_vec

def tfidf_char_ngrams(text, dataset, n):
    t_c_ngrams = TfidfVectorizer(analyzer = "char", ngram_range = (n,n))
    t_c_ngrams_vec = t_c_ngrams.fit_transform(dataset).toarray()
    text_vec = t_c_ngrams_vec[np.where(dataset == text)]
    
    return text_vec.T, t_c_ngrams_vec

start = time.time()
one = count_word_ngrams(text_uniques[0], text_uniques[0:100], 3)
#two = count_char_ngrams(text_uniques[0], text_uniques, 3)
#three = tfidf_word_ngrams(text_uniques[0], text_uniques, 3)
#four = tfidf_char_ngrams(text_uniques[0], text_uniques, 3)
end = time.time()

print(end - start)