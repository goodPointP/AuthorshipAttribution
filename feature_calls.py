from functions import *
from feature_extraction_seb import *
from readability import Readability
import pandas as pd
#%%

rawData = read_data()
rawTruths = read_truth_data()
data = textimport_light(rawData)
#%%
idx, corpus = remove_duplicates(data[:100])

#%%
def preprocessing_complete(corpus):
    corpus = list(corpus)
    minus_punc = remove_punc(corpus)
    minus_stop = remove_stop(corpus)
    minus_both = remove_punc_stop(corpus)
    return [corpus, minus_punc, minus_stop, minus_both]

#%%
corpora = preprocessing_complete(corpus)                                #0.8s / 100 texts
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
    
    # corpora[2] = stopwords removed
    pr = punctuation_ratio(corpora[2])                                  #0.3s
    sc = special_characters(corpora[2])                                 #0.2s
    
    # corpora[3] = both removed
    iw = intensifier_words(corpora[3])                                  #0.08s
    ta = time_adverbs(corpora[3])                                       #0.09s
    dgt = digits(corpora[3])                                            #0.2s
    return list(zip(*(asl, awl, fwf, fws, hl, liwc, rm_s, rm_wt, rm_avg_s, ttr, pr, sc, iw, ta, dgt))) 
    


#%%

test = arrays_combined(corpora)
    
#%%


def readability_metrics2(corpus):
    
    scores = []
    sentence_beginnings = []
    word_types = []
    avg_syllables = []
    helper = []
    for text in corpus:
       # line_sep = '.\n'.join(sentences)
        readability_results = Readability(text)
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
        
    return helper #scores, avg_syllables #word_types

#%%

text = corpora[1][0]
readability_results = readability.getmeasures(text, lang='en') #21ms
w_classes = ['auxverb', 'conjunction','nominalization',
                     'preposition','pronoun', 'tobeverb' ]
   
begin_types = ['article','conjunction','interrogative'
               ,'preposition','pronoun','subordination']
#%%
%%time
f = readability_metrics2(corpora[1])

        

