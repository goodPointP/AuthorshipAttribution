from nltk.tokenize import sent_tokenize, RegexpTokenizer
from LPfunctions import textimport_light
import string
import numpy as np

df = textimport_light()

txt = df.loc[55][2][0]

def tokenize(text):
    word_tokenizer = RegexpTokenizer(r"\w+")
    w = word_tokenizer.tokenize(text)
    s = sent_tokenize(text)
    return w,s

words, sentences = tokenize(txt)
    
def avg_word_length(text):
    average_lettercount = sum(len(word) for word in words) / len(words)
    return average_lettercount

def avg_sentence_length(text):
    average_wordcount = len(words) / len(sentences)
    return average_wordcount

def punctuation(text):
    count = 0
    for i in text:   
        if i in string.punctuation:  
            count += 1 
    return count
    
def function_words(text):
    f_string = []
    with open('data/function_words.txt') as f:
        for l in f:
           f_string.append(l)
    f_words = tokenize(f_string[0])[0]
    f_word_freq = [words.count(f_word) for f_word in f_words]
    
    return dict(list(zip(f_words,f_word_freq)))

def intensifier_words(text):
    i_words = []
    with open('data/intensifier_words.txt') as i:
        for w in i:
           i_words.append(w)
    i_word_freq = [words.count(i_word) for i_word in i_words]
    
    return dict(list(zip(i_words,i_word_freq)))
        
def TTR(text):
    return (len(set(words)) / (len(words)))*100


    

