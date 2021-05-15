from functions import *
from feature_extraction import *
from ngrams import *
from POS import pos_tag, skipgramming
from scipy import spatial
from sklearn.metrics.pairwise import cosine_distances
import textstat #pip install textstat
import pandas as pd
import time
import readability
from multiprocessing import Pool
import psutil
import pickle

#%%
raw_data = read_data()
df = textimport_light(raw_data, pandas_check = True)
text_IDs, text_uniques = remove_duplicates(df)
df['text_id'] = pd.Series(zip(text_IDs[0::2], text_IDs[1::2]))

#%%
start = time.time()
batch_size = 100
corpora = preprocessing_complete(text_uniques[0:batch_size])  

with open('data/pos_tags_whole_text_4536.pkl', 'rb') as f:
    pos = pickle.load(f)
    
num_pairs = int(batch_size/2)
end = time.time()

print(f"Execution time was {end-start}s")

#%%

def calls(corpora):                                           #9.5s / 100 texts
    
    # corpora[0] = no preprocessing
    asl = avg_sentence_length(corpora[0])                               #0.5s
    
    # corpora[1] = punctuation removed
    awl = avg_word_length(corpora[1])                                   #0.07s
    fwf, fws = function_words(corpora[1])                               #0.8s
    hl = hapax_legomena(corpora[1])                                     #0.5s
    liwc = LIWC(corpora[1])                                             #1.9s
    fl_kinc, avg_syll = readability_metrics(corpora[1])                 #1.45s
    ttr = TTR(corpora[1])                                               #0.1s
    w_tg = tfidf_word_ngrams(corpora[1], 3,3)
    c_tg = tfidf_char_ngrams(corpora[1], 3,3)
    i_o_c = index_of_coincidence(corpora[1])                            #0.06s
    
    # corpora[2] = stopwords removed
    pr = punctuation_ratio(corpora[2])                                  #0.3s
    sc = special_characters(corpora[2])                                 #0.2s
    
    # corpora[3] = both removed
    iw = intensifier_words(corpora[3])                                  #0.08s
    ta = time_adverbs(corpora[3])                                       #0.09s
    dgt = digits(corpora[3])
    
    #pos_tags as input
    st = skipgramming(pos)
    adj_adv = adj_adv_ratio(pos)
    si_t = simple_tense(pos)
    
    #returns one list for float output, one for lists, and one where the output is already sim or cos
    return list(zip(*(asl, awl, fws, hl, ttr, pr, sc, dgt, adj_adv, i_o_c, fl_kinc, avg_syll))), list(zip(*(fwf, liwc, fwf, iw, ta, si_t))), list(zip(*(w_tg, c_tg, st)))


#%% 26.5s for 100 texts...

def arrays_combined(corpora):
    
    floats, arrays, sims = calls(corpora)
    
    floats_matrix = np.array([float_ for float_ in floats], dtype=object)
    tagged_pairs_one = floats_matrix[np.array(list(df['text_id'][:num_pairs]))]
    distance_matrix = np.stack([dist(pair[0], pair[1]) for pair in tagged_pairs_one])
    
    array_matrix = np.array([array_ for array_ in arrays], dtype = object)
    tagged_pairs_two = array_matrix[np.array(list(df['text_id'][:num_pairs]))]
    tagged_pairs_split = [p for p in tagged_pairs_two]
    
    cos = []
    
    for pair in tagged_pairs_split:
        featuresFirst = [feature for feature in pair[0]]
        featuresSecond = [feature for feature in pair[1]]
    
        for i, feature in enumerate(pair[0]):
            feature_vector_1 = np.array(featuresFirst[i]).reshape(1,-1)
            feature_vector_2 = np.array(featuresSecond[i]).reshape(1,-1)
            cos.append(float(cosine(feature_vector_1, feature_vector_2)))
    
    cos_matrix = np.stack(np.array_split(np.array((cos)), num_pairs))
    
    sim_matrix = np.stack(sims)
    feature_matrix = np.hstack((distance_matrix, cos_matrix, sim_matrix))

    return feature_matrix

#%%
feature_matrix = arrays_combined(corpora)

with open('feature_matrix.pkl', 'wb') as f:
     pickle.dump(feature_matrix,f)
