from functions import *
from POS import pos_tag, skipgramming
from nltk.probability import FreqDist
import string
import readability # pip install readability # #https://github.com/mmautner/readability/blob/master/readability.py
import pandas as pd
import numpy as np
from scipy import spatial
import liwc
import re
import textstat
from collections import Counter

#%%
# RUN ONLY ONCE
# read the data
if __name__ == '__main__':
    rawData = read_data()
    rawTruths = read_truth_data()
    data = textimport_light(rawData)

#%%

#punc: yes
#stop: yes

def avg_sentence_length(corpus):
    s_lengths = []
    for text in corpus:
        sents = [remove_punctuation(sent) for sent in text.split(". ")]
        s_lengths.append(len(''.join(sents).split())/len(sents))
    return s_lengths

#punc: yes
#stop: no

def punctuation_ratio(corpus):
    ratios = []
    for text in corpus:   
        count = sum(c in string.punctuation for c in text)
        ratios.append(count/len(text.split()))
    return ratios

def special_characters(corpus):
    spec = '#$%&\()*+-/<=>@[\\]^_{|}~'
    spec_count = []
    for text in corpus:
        spec_count.append(sum(c in spec for c in text.replace(" ", ""))/len(text))
    return spec_count

#punc: no
#stop: yes

def avg_word_length(corpus):
    lengths = []
    for text in corpus:
        lengths.append(len(text.replace(" ", "")) / len(text.split()))
    return lengths

# index 0 returns frequency count / text lenght and index 1 returns sum / text lenght
def function_words(corpus):
    f_string = []
    f_word_sum = []
    f_word_freq = []
    with open('data/function_words.txt') as f:
        for l in f:
            f_string.append(l.strip())
    f_words = f_string[0].split()
    for text in corpus:
        freq = [text.count(f_word) for f_word in f_words]
        freq_ratio = [i / len(text) for i in freq]
        f_word_freq.append(freq_ratio)
        f_word_sum.append(sum(freq) / len(text))
    return f_word_freq, f_word_sum

def TTR(corpus):
    ttr = []
    for text in corpus:
        ttr.append((len(set(text.split())) / (len(text.split())))*100)
    return ttr

def hapax_legomena(corpus):
    haps = []
    for text in corpus:
        haps.append(len(FreqDist(text.split()).hapaxes()) / len(text))
    return haps

### Index to get different measures
def readability_metrics(corpus):
    
    flesch_kincaid = []
    avg_syllables = []
    avg_char_per_word = []
    avg_difficult_words = []

    for text in corpus:
        flesch_kincaid.append(textstat.flesch_kincaid_grade(text))
        avg_syllables.append(textstat.avg_syllables_per_word(text))
        
        # very slow!
        # avg_difficult_words.append(textstat.difficult_words(text)/len(text.split()))

    return flesch_kincaid, avg_syllables#, avg_difficult_words


def LIWC(corpus):

    def liwc_tokenize(text):
        for match in re.finditer(r'\w+', text, re.UNICODE):
            yield match.group(0)
    
    liwc_vecs = []
    parse, category_names = liwc.load_token_parser('data/LIWC2007_English100131.dic')
    
    for text in corpus:
        token_text = liwc_tokenize(text)
        counterVar = Counter()
        counterVar.update({x:0 for x in category_names})
        counterVar.update(category for token in token_text for category in parse(token))
        liwc_vec = list(dict(counterVar).values())
        liwc_ratio = [l / len(text) for l in liwc_vec]
        liwc_vecs.append(liwc_ratio)
        
    return liwc_vecs

#punc: no
#stop: no

def intensifier_words(corpus):
    i_words = []
    i_word_freqs = []
    with open('data/intensifier_words.txt') as i:
        for l in i:
           i_words.append(l.strip())
    for text in corpus:
        i_word_freq = [text.count(i_word) for i_word in i_words]
        i_word_ratio = [f / len(text) for f in i_word_freq]
        i_word_freqs.append(i_word_ratio)
    return i_word_freqs

def time_adverbs(corpus):
    t_words = []
    t_word_freqs = []
    with open('data/time_adverbs.txt') as t:
        for l in t:
           t_words.append(l.strip())
    for text in corpus:
        t_word_freq = [text.count(t_word) for t_word in t_words]
        t_word_ratio = [t / len(text) for t in t_word_freq]
        t_word_freqs.append(t_word_ratio)
        
    return t_word_freqs

def digits(corpus):
    digits = []
    for text in corpus:
        digits.append(sum(c.isnumeric() for c in text) / len(text))
    return digits

def index_of_coincidence(corpus):
    normalizing_coef = 26.0 #26 for english
    ioc = []
    for text in corpus:
        text = remove_spaces(text)
        text = text.upper()
        
        frequencySum = 0.0
        for letter in string.ascii_uppercase:
            frequencySum += float(text.count(letter)) * (float(text.count(letter))-1)
            
        N = len(text)
        ioc.append(frequencySum / (N*(N-1)) ) #* (normalizing_coef/(N*(N-1)))
    return ioc

#%%
#These functions take POS tags, not corpora as input

def adj_adv_ratio(pos_tags):
    
    adj_adv = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS"]
    adj_adv_rat = []
    for i, j in enumerate(pos_tags):
        co = Counter()
        co.update({x:pos_tags[i].count(x) for x in adj_adv})
        ratio = sum(list(dict(co).values())) / len(pos_tags[i])
        adj_adv_rat.append(ratio)
        
    return adj_adv_rat

#This is a very simple approach but a more elaborate one would require advanced parsing
def simple_tense(pos_tags):
    tense_ratio = []
    for i, j in enumerate(pos_tags):
        present = len([word for word in pos_tags[i] if word in ["VBG", "VBP", "VBZ"]]) / len(pos_tags[i])
        past = len([word for word in pos_tags[i] if word in ["VBD", "VBN"]]) / len(pos_tags[i])
        tense = [present, past]
        tense_ratio.append(tense)
    return tense_ratio

#%%

# # with open ('data2289-withFeatures.pkl', 'wb') as f:
# #     pickle.dump(data,f)

# # with open('data2289-withFeatures.pkl', 'rb') as f:
# #    mynewlist = pickle.load(f)
