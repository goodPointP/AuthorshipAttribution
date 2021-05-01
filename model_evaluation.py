from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from functions import *
import numpy as np
from feature_extraction import create_input_matrix
import liwc
import re
from collections import Counter

#%%
rawData = read_data()
data = textimport_light(rawData, True)
rawTruth = read_truth_data()
truths = truthimport_light(rawTruth)
labels = truths['same'] 

#%%
#Text pairs to include
samples = int(1000)

#deal with Nan values, normalize data
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
input_matrix = normalize(imputer.fit_transform(create_input_matrix(data[0:samples])))

#split into train and test sets
X_train, X_test, y_train, y_test = split_data(input_matrix, labels[0:samples])

#%%
pcp = Perceptron()

pcp.fit(X_train, y_train)
pcp_pred = pcp.predict(X_test)
pcp_acc = accuracy_score(y_test, pcp_pred)

#%%
lg = LogisticRegression()

lg.fit(X_train, y_train)
lg_pred = lg.predict(X_test)
lg_acc = accuracy_score(y_test, lg_pred)

#%%
svm = LinearSVC()

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

#%%
rf =  RandomForestClassifier()

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

#%%

def LIWC(text):

    def liwc_tokenize(text):
        for match in re.finditer(r'\w+', text, re.UNICODE):
            yield match.group(0)

    parse, category_names = liwc.load_token_parser('data/LIWC2007_English100131.dic')
    token_text = liwc_tokenize(text)
    count = dict(Counter(category for token in token_text for category in parse(token)))
    #liwc_dict = list()).values())
    #liwc_vec = np.array(liwc_count)
    return count

zzz = LIWC(data.loc[18][2][0])

