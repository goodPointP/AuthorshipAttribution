from preprocessing import *
from nltk.probability import FreqDist
import string
import readability # pip install readability # #https://github.com/mmautner/readability/blob/master/readability.py
import pandas as pd
import numpy as np
from scipy import spatial

    
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


def combined(text, words, sentences):
        
    int_or_float = np.array([avg_word_length(words),
                     avg_sentence_length(words, sentences), punctuation_ratio(text, words), 
                     TTR(words), hapax_legomena(words), readability_metrics(sentences)[0],
                     readability_metrics(sentences)[3],
                     digits(words), superlatives(words),
                     special_characters(words)], dtype="object")
    
    list_or_array = np.array([function_words(words), intensifier_words(words), 
                    time_adverbs(words), readability_metrics(sentences)[1],
                    readability_metrics(sentences)[2]], dtype="object")
    
    return int_or_float, list_or_array


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
        vector = vectorize(data_frame.loc[i][2][0], df.loc[i][2][1])
        vectors.append(vector)
    
    matrix = np.stack(vectors, axis = 1)

    return matrix
        
df = textimport_light()[0:10]
matrix = create_input_matrix(df)

#def exact_word_matches(words, words2):
 #   unique1 = set(words)
  #  unique2 = set(words2)
   # matches = len(unique1.intersection(unique2))
    #return matches
    

    