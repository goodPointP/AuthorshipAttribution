from LPfunctions import textimport_light, tokenizer
from nltk.probability import FreqDist
import string
import readability # pip install readability

df = textimport_light()
text = df.loc[26][2][0]
words, sentences = tokenizer(text)
line_sep = '.\n'.join(sentences)
readability_results = readability.getmeasures(line_sep, lang='en')
    
def avg_word_length(text):
    average_lettercount = sum(len(word) for word in words) / len(words)
    
    return average_lettercount

def avg_sentence_length():
    average_wordcount = len(words) / len(sentences)
    
    return average_wordcount

def punctuation_ratio():
    count = 0
    for i in text:   
        if i in string.punctuation:  
            count += 1 
            
    return count / len(words)
    
def function_words():
    f_string = []
    with open('data/function_words.txt') as f:
        for l in f:
           f_string.append(l)
    f_words = tokenizer(f_string[0])[0]
    f_word_freq = [words.count(f_word) for f_word in f_words]
    
    return dict(list(zip(f_words,f_word_freq)))

fw = function_words()

def intensifier_words():
    i_words = []
    with open('data/intensifier_words.txt') as i:
        for l in i:
           i_words.append(l)
    i_word_freq = [words.count(i_word) for i_word in i_words]
    
    return dict(list(zip(i_words,i_word_freq)))

def time_adverbs():
    t_words = []
    with open('data/time_adverbs.txt') as t:
        for l in t:
           t_words.append(l)
    t_word_freq = [words.count(t_word) for t_word in t_words]
    
    return dict(list(zip(t_words,t_word_freq)))
        
def TTR(text):
    
    return (len(set(words)) / (len(words)))*100


def hapax_legomena():
    f_dist = FreqDist(words)
    hapax = f_dist.hapaxes()
    
    return len(hapax)

#https://github.com/mmautner/readability/blob/master/readability.py

def readability_score():
    
    return readability_results['readability grades']['Kincaid']

def sentence_beginnings():
    
    return [
        readability_results['sentence beginnings']['pronoun'],
        readability_results['sentence beginnings']['interrogative'],
        readability_results['sentence beginnings']['article'],
        readability_results['sentence beginnings']['subordination'],
        readability_results['sentence beginnings']['conjunction'],
        readability_results['sentence beginnings']['preposition']
        ]

def word_usage():
    
    return [
        readability_results['word usage']['auxverb'],
        readability_results['word usage']['conjunction'],
        readability_results['word usage']['nominalization'],
        readability_results['word usage']['preposition'],
        readability_results['word usage']['pronoun'],
        readability_results['word usage']['tobeverb']
        ]

def avg_syllables_per_word():
    
    return  readability_results['sentence info']['syll_per_word']
    