from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, HalvingGridSearchCV
from functions import *
import numpy as np

#%%

with open('feature_matrix.pkl', 'rb') as f:
    feat_matrix = pickle.load(f)

rawTruth = read_truth_data()
truths = truthimport(rawTruth)
labels = truths['same'] 
results = []
#%%

#deal with Nan values, normalize data #split into train and test sets
cv = RepeatedStratifiedKFold(n_splits=5)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
input_matrix = normalize(imputer.fit_transform(feat_matrix))
X_train, X_test, y_train, y_test = split_data(input_matrix, labels)


#%%
lg = LogisticRegression(C=0.01, penalty='l2', solver='newton-cg', max_iter=1000)
#lg = LogisticRegression()

lg.fit(X_train, y_train)
lg_pred = lg.predict(X_test)
lg_acc = metrics.accuracy_score(y_test, lg_pred)
lg_auc = metrics.roc_auc_score(y_test, lg_pred)
lg_rep = metrics.classification_report(y_test, lg_pred)

#%%
svm = SVC(C=5, gamma=0.01, kernel='rbf')
#svm = SVC()

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = metrics.accuracy_score(y_test, svm_pred)
svm_auc = metrics.roc_auc_score(y_test, svm_pred)
svm_rep = metrics.classification_report(y_test, svm_pred)

#%%

rf = RandomForestClassifier(max_depth=90, max_features=2, min_samples_leaf=3, min_samples_split=10, n_estimators=200)
#rf = RandomForestClassifier()

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = metrics.accuracy_score(y_test, rf_pred)
rf_auc = metrics.roc_auc_score(y_test, rf_pred)
rf_rep = metrics.classification_report(y_test, rf_pred)

#%%
mlp = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='constant', solver='adam', max_iter=1000)
#mlp = MLPClassifier(max_iter=1000)

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = metrics.accuracy_score(y_test, mlp_pred)
mlp_auc = metrics.roc_auc_score(y_test, mlp_pred)
mlp_rep = metrics.classification_report(y_test, mlp_pred)

#%%#Leave-one_out method to see which features contribute 

# acc_scores = []
# for i in range(len(feat_matrix.T)):
#     print(i)
#     lg_i = LogisticRegression()
#     X_train_i = np.delete(X_train, obj = i, axis = 1 )
#     X_test_i = np.delete(X_test, obj = i, axis = 1 )
#     lg_i.fit(X_train_i, y_train)
#     lg_pred_i = lg_i.predict(X_test_i)
#     lg_acc_i = metrics.accuracy_score(y_test, lg_pred_i)
#     acc_scores.append(lg_acc_i)
