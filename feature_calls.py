from functions import *
from feature_extraction_seb import *
from ngrams import *
from pos_function2 import skipgramming
from scipy import spatial
from sklearn.metrics.pairwise import cosine_distances
#from readability import Readability
import textstat #pip install textstat
import pandas as pd

#%%

raw_data = read_data()
df = textimport_light(raw_data, pandas_check = True)
text_IDs, text_uniques = remove_duplicates(df)
df['text_id'] = pd.Series(zip(text_IDs[0::2], text_IDs[1::2]))

#%%
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

#%%
corpora = preprocessing_complete(text_uniques[0:100])                   #0.8s / 100 texts
#%%

def arrays_combined(corpora):                                           #9.5s / 100 texts
    
    # corpora[0] = no preprocessing
    asl = avg_sentence_length(corpora[0])                               #0.5s
    
    # corpora[1] = punctuation removed
    awl = avg_word_length(corpora[1])                                   #0.07s
    fwf, fws = function_words(corpora[1])                               #0.8s
    hl = hapax_legomena(corpora[1])                                     #0.5s
    liwc = LIWC(corpora[1])                                             #1.9s
    rm_s, rm_wt, rm_avg_s = readability_metrics(corpora[1])             #4.6s
    ttr = TTR(corpora[1])                                               #0.1s
    w_tg = tfidf_word_ngrams(corpora[1], 3,3)
    c_tg = tfidf_char_ngrams(corpora[1], 3,3)
    
    # corpora[2] = stopwords removed
    pr = punctuation_ratio(corpora[2])                                  #0.3s
    sc = special_characters(corpora[2])                                 #0.2s
    
    # corpora[3] = both removed
    iw = intensifier_words(corpora[3])                                  #0.08s
    ta = time_adverbs(corpora[3])                                       #0.09s
    dgt = digits(corpora[3])
    
    #returns one list for float output, one for lists, and one where the output is already sim or cos
    return list(zip(*(asl, awl, fws, hl, ttr, pr, sc, dgt))), list(zip(fwf, liwc, fwf, iw, ta)), list([w_tg, c_tg])
            

#%%

floats, arrays, sims = arrays_combined(corpora)
    
#%%
floats_array = np.array([float_ for float_ in floats], dtype=object)
tagged_pairs_one = floats_array[np.array(list(df['text_id'][:int(len(corpora[0])/2)]))] 
sim = [dist(pair[0], pair[1]) for pair in tagged_pairs_one]

array_array = np.array([array_ for array_ in arrays])
tagged_pairs_two = array_array[np.array(list(df['text_id'][:int(len(corpora[0])/2)]))] 
tagged_pairs_split = [p for p in tagged_pairs_two]
cos = [cosine(np.ndarray(pair[0], dtype="float"), np.ndarray(pair[1], dtype="float")) for pair in tagged_pairs_split]


#%%
def readability_metrics2(corpus):
    
    scores = []
    sentence_beginnings = []
    word_types = []
    avg_syllables = []
    helper = []
    for text in corpus:
       # line_sep = '.\n'.join(sentences)
        # readability_results = Readability(text)
        # scores.append(Readability(text).flesch_kincaid())
        scores.append(textstat.flesch_kincaid_grade(text))
        # begin_types = ['article','conjunction','interrogative'
        #                ,'preposition','pronoun','subordination']
        
        # w_classes = ['auxverb', 'conjunction','nominalization',
        #              'preposition','pronoun', 'tobeverb' ]
        
        # red_score = readability_results['readability grades']['Kincaid']
        # scores.append(red_score)
        
        # types = [readability_results['word usage'][i] for i in w_classes]
        # types_ratio = [r / len(text) for r in types]
        # word_types.append(types_ratio)
        
        # syllables = [readability_results['sentence info']['syll_per_word']]
        # avg_syllables.append(syllables)
        
    return  scores#, avg_syllables #word_types

#%%

text = corpora[1][0]
# readability_results = readability.getmeasures(text, lang='en') #21ms
w_classes = ['auxverb', 'conjunction','nominalization',
                     'preposition','pronoun', 'tobeverb' ]
   
begin_types = ['article','conjunction','interrogative'
               ,'preposition','pronoun','subordination']

#%%
%%time
f = readability_metrics2(corpora[1])

        

