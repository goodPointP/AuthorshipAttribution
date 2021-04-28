### POS-tagging.py #0s
import pandas as pd
import numpy as np
from functions import textimport_light, truthimport_light
from collections import Counter
import spacy

#%% Creating our corpus of texts. Labels and IDs are not included. #1.5s

print("first part")
df = textimport_light()
df_truth = truthimport_light()


#%% For splitting the textpairs into an array of 1xN. #0.2s


text_list = pd.Series(df['pair'].explode())
text_IDs, text_uniques = text_list.factorize()
df['text_id'] = pd.Series(zip(text_IDs[0::2], text_IDs[1::2]))

#%% defining the subset used (no. of texts)

no = int(50)
texts = ['"{}"'.format(str(text.replace('"', "'"))) for text in text_uniques[:no]]

#%% initializing spacy and excluding redundancy. #extracting POS-tags from texts.
nlp = spacy.load("en_core_web_sm", exclude=["parser", "senter","ner"])

tags = []
for doc in nlp.pipe(texts, batch_size=50):
    tags.append([token.tag_ for token in doc])
    
#%% recombining the pairs of texts as pairs of POS-tags. #Counting POS-tags.

tagged_pairs = np.array(tags, dtype=object)[np.array(tuple(df['text_id'][:int(no/2)]))]
counts = [Counter(tag for tag in text) for text in tags]
