from functions import *
from nltk.probability import FreqDist
import string
import readability # pip install readability # #https://github.com/mmautner/readability/blob/master/readability.py
import pandas as pd
import numpy as np
from scipy import spatial
import liwc
import re
from collections import Counter

#%%
# RUN ONLY ONCE
# read the data
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

### Fixed but horribly slow and I don't know how to make more efficient
### Index to get different measures
def readability_metrics(corpus):
    
    scores = []
    sentence_beginnings = []
    word_types = []
    avg_syllables = []
    
    for text in corpus:
       # line_sep = '.\n'.join(sentences)
        readability_results = readability.getmeasures(text, lang='en')
        
        begin_types = ['article','conjunction','interrogative'
                       ,'preposition','pronoun','subordination']
        
        w_classes = ['auxverb', 'conjunction','nominalization',
                     'preposition','pronoun', 'tobeverb' ]
        
        red_score = readability_results['readability grades']['Kincaid']
        scores.append(red_score)
        
        types = [readability_results['word usage'][i] for i in w_classes]
        types_ratio = [r / len(text) for r in types]
        word_types.append(types_ratio)
        
        syllables = [readability_results['sentence info']['syll_per_word']]
        avg_syllables.append(syllables)
        
    return scores, word_types, avg_syllables



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

def index_of_coincidence(text):
    # TODO: REMOVE PUNCTUATION
    normalizing_coef = 26.0 #26 for english
    text = remove_punctuation(text)
    text = remove_spaces(text)
    text = text.upper()
    
    N = len(text)
    
    # chances = []
    frequencySum = 0.0
    for letter in string.ascii_uppercase:
        frequencySum += float(text.count(letter)) * (float(text.count(letter))-1)
        # chances.append( (text.count(letter) * (text.count(letter)-1))/(len(text) * (len(text)-1)) )
    
    # print(chances)
    # print(sum(chances))
    # ioc = sum(chances)/((len(text) * (len(text)-1)))
    ioc = frequencySum / (N*(N-1)) #* (normalizing_coef/(N*(N-1)))
    return ioc

# #%%

# def combined(text, words, sentences):

#     int_or_float = np.array([avg_word_length(words),
#                      avg_sentence_length(words, sentences), punctuation_ratio(text, words), 
#                      TTR(words), hapax_legomena(words), readability_metrics(sentences)[0],
#                      readability_metrics(sentences)[3],
#                      digits(words), superlatives(words),
#                      special_characters(words), index_of_coincidence(text)], dtype="object")
    
#     list_or_array = np.array([function_words(words), intensifier_words(words), 
#                     time_adverbs(words), readability_metrics(sentences)[1],
#                     readability_metrics(sentences)[2], LIWC(text)], dtype="object")
    
#     return [int_or_float, list_or_array]


# def vectorize(text1, text2):
    
#     def difference(num1, num2):
#         dif = np.subtract(max(num1, num2), min(num1, num2))
#         return dif

#     def cosine_sim(vec1, vec2):
#         sim = spatial.distance.cosine(vec1, vec2)
#         return sim
    
#     words1, sentences1 = tokenizer(text1)
#     words2, sentences2 = tokenizer(text2)
    
#     nums1 = combined(text1, words1, sentences1)[0]
#     vecs1 = combined(text1, words1, sentences1)[1]
#     nums2 = combined(text2, words2, sentences2)[0]
#     vecs2 = combined(text2, words2, sentences2)[1]

#     feat_vec_part_one = np.zeros(len(nums1))
#     feat_vec_part_two = np.zeros(len(vecs1))
    
#     for i in range(len(nums1)):
#         feat_vec_part_one[i] = difference(nums1[i], nums2[i])

#     for i in range(len(vecs1)):
#         feat_vec_part_two[i] = cosine_sim(vecs1[i], vecs2[i])

        
#     feat_vec = np.hstack([feat_vec_part_one, feat_vec_part_two])
    
#     return feat_vec

# def create_input_matrix(data_frame):
    
#     vectors = []
#     for i in range(len(data_frame)):
#         vector = vectorize(data_frame.loc[i][2][0], data_frame.loc[i][2][1])
#         vectors.append(vector)
    
#     matrix = np.stack(vectors, axis = 1)

#     return matrix.T
        
# data = textimport_light(rawData, True)
# matrix = create_input_matrix(data[0:100])

# def getFeatures(text):
#     return features

# #for index, row in enumerate(data[:10]):
#  #   text1 = data[index]['pair'][0]
#   #  text2 = data[index]['pair'][1]
    
#    # words1, sentences1 = tokenizer(text1)
#    # words2, sentences2 = tokenizer(text2)
    
#    # data[index]['features'] = [np.hstack((combined(text1, words1, sentences1))), np.hstack((combined(text2, words2, sentences2)))]

# # with open ('data2289-withFeatures.pkl', 'wb') as f:
# #     pickle.dump(data,f)

# # with open('data2289-withFeatures.pkl', 'rb') as f:
# #    mynewlist = pickle.load(f)
