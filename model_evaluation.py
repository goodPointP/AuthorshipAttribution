from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from functions import *
import numpy as np
from feature_extraction import create_input_matrix

#%%

rawData = read_data()
data = textimport_light(rawData, True)
rawTruth = read_truth_data()
truths = truthimport_light(rawTruth)
labels = truths['same'] 

#%%
#deal with Nan values, normalize data and split into train and test sets

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
input_matrix = normalize(imputer.fit_transform(create_input_matrix(data[0:100])))

X_train, X_test, y_train, y_test = split_data(input_matrix, labels[0:100])

#%%
pcp = Perceptron()

pcp.fit(X_train, y_train)
pred = pcp.predict(X_test)
acc = accuracy_score(y_test, pred)

#%%


