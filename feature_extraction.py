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

#%%
def avg_word_length(words):
    return sum(len(word) for word in words) / len(words)

def avg_sentence_length(words, sentences):
    return len(words) / len(sentences)

def punctuation_ratio(text, words):
    count = 0
    for i in text:   
        if i in string.punctuation:  
            count += 1    
    return count / len(words)
    
def function_words(words):
    f_string = []
    with open('data/function_words.txt') as f:
        for l in f:
           f_string.append(l)
    f_words = tokenizer(f_string[0])[0]
    f_word_freq = [words.count(f_word) for f_word in f_words]
    return f_word_freq

def intensifier_words(words):
    i_words = []
    with open('data/intensifier_words.txt') as i:
        for l in i:
           i_words.append(l)
    i_word_freq = [words.count(i_word) for i_word in i_words]
    return i_word_freq

def time_adverbs(words):
    t_words = []
    with open('data/time_adverbs.txt') as t:
        for l in t:
           t_words.append(l)
    t_word_freq = [words.count(t_word) for t_word in t_words]
    return t_word_freq
        
def TTR(words):
    return (len(set(words)) / (len(words)))*100

def hapax_legomena(words):
    f_dist = FreqDist(words)
    hapax = f_dist.hapaxes()
    return len(hapax)

def readability_metrics(sentences):
    line_sep = '.\n'.join(sentences)
    readability_results = readability.getmeasures(line_sep, lang='en')
    red_score = readability_results['readability grades']['Kincaid']
    begin_types = ['article','conjunction','interrogative'
                   ,'preposition','pronoun','subordination']
    
    beginnings = [readability_results['sentence beginnings'][i] for i in begin_types]

    w_classes = ['auxverb', 'conjunction','nominalization',
                  'preposition','pronoun', 'tobeverb' ]
    
    word_types = [readability_results['word usage'][i] for i in w_classes]
    avg_syllables = [readability_results['sentence info']['syll_per_word']]
    
    return red_score, beginnings, word_types, avg_syllables

def digits(words):
    digits = 0
    for word in words:
        for letter in word:
            if letter.isnumeric():
                digits += 1 
    return digits

def superlatives(words):
    suffix = 'est'
    sup = 0
    for word in words:
        if word.endswith(suffix):
            sup += 1
        elif word == 'most':
            sup += 1
    return sup

def special_characters(words):
    spec = ['~', '@', '#', '$', '%', '^', '&', '*', '-',
            '_', '=', '+', '>', '<', '[',']', '{', '}','/',]
    spec_count = 0
    for word in words:
        for letter in word:
            if letter in spec:
                spec_count += 1
    return spec_count

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

def LIWC(text):

    def liwc_tokenize(text):
        for match in re.finditer(r'\w+', text, re.UNICODE):
            yield match.group(0)

    parse, category_names = liwc.load_token_parser('data/LIWC2007_English100131.dic')
    token_text = liwc_tokenize(text)
    liwc_vec = list(dict(Counter(category for token in token_text for category in parse(token))).values())
    
    return liwc_vec

#%%

def combined(text, words, sentences):
    # print(text)
    int_or_float = np.array([avg_word_length(words),
                     avg_sentence_length(words, sentences), punctuation_ratio(text, words), 
                     TTR(words), hapax_legomena(words), readability_metrics(sentences)[0],
                     readability_metrics(sentences)[3],
                     digits(words), superlatives(words),
                     special_characters(words), index_of_coincidence(text)], dtype="object")
    
    list_or_array = np.array([function_words(words), intensifier_words(words), 
                    time_adverbs(words), readability_metrics(sentences)[1],
                    readability_metrics(sentences)[2], LIWC(text)], dtype="object")
    
    return [int_or_float, list_or_array]


def vectorize(text1, text2):
    
    def difference(num1, num2):
        dif = np.subtract(max(num1, num2), min(num1, num2))
        return dif

    def cosine_sim(vec1, vec2):
        sim = spatial.distance.cosine(vec1, vec2)
        return sim
    
    words1, sentences1 = tokenizer(text1)
    words2, sentences2 = tokenizer(text2)
    
    nums1 = combined(text1, words1, sentences1)[0]
    vecs1 = combined(text1, words1, sentences1)[1]
    nums2 = combined(text2, words2, sentences2)[0]
    vecs2 = combined(text2, words2, sentences2)[1]

    feat_vec_part_one = np.zeros(len(nums1))
    feat_vec_part_two = np.zeros(len(vecs1))
    
    for i in range(len(nums1)):
        feat_vec_part_one[i] = difference(nums1[i], nums2[i])

    for i in range(len(vecs1)):
        feat_vec_part_two[i] = cosine_sim(vecs1[i], vecs2[i])

        
    feat_vec = np.hstack([feat_vec_part_one, feat_vec_part_two])
    
    return feat_vec

def create_input_matrix(data_frame):
    
    vectors = []
    for i in range(len(data_frame)):
        vector = vectorize(data_frame.loc[i][2][0], data_frame.loc[i][2][1])
        vectors.append(vector)
    
    matrix = np.stack(vectors, axis = 1)

    return matrix.T
        
data = textimport_light(rawData, True)
matrix = create_input_matrix(data[0:100])

def getFeatures(text):
    return features

#for index, row in enumerate(data[:10]):
 #   text1 = data[index]['pair'][0]
  #  text2 = data[index]['pair'][1]
    
   # words1, sentences1 = tokenizer(text1)
   # words2, sentences2 = tokenizer(text2)
    
   # data[index]['features'] = [np.hstack((combined(text1, words1, sentences1))), np.hstack((combined(text2, words2, sentences2)))]

# with open ('data2289-withFeatures.pkl', 'wb') as f:
#     pickle.dump(data,f)

# with open('data2289-withFeatures.pkl', 'rb') as f:
#    mynewlist = pickle.load(f)

#%%
X = []
# for index, row in enumerate(data):
    # X.append(data[index]['features'])

# Y = []
# Y = labels['same']

#%%
