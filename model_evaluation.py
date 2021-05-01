from sklearn.linear_model import Perceptron
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from functions import *
from feature_extraction import create_input_matrix

rawData = read_data()
data = textimport_light(rawData, True)

#%%

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
input_matrix = imputer.fit_transform(create_input_matrix(data[0:100]).T)
X_train, X_test, y_train, y_test = split_data(input_matrix, labels[0:100])
truths = truthimport_light(rawTruths)
labels = truths['same']

pcp = Perceptron()

pcp.fit(X_train, y_train)
pred = pcp.predict(X_test)
acc = accuracy_score(y_test, pred)

