from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

#%%

#deal with Nan values, normalize data #split into train and test sets
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
input_matrix = normalize(imputer.fit_transform(feat_matrix))
X_train, X_test, y_train, y_test = split_data(input_matrix, labels[0:len(input_matrix)])


#%%
lg = LogisticRegression()

lg.fit(X_train, y_train)
lg_pred = lg.predict(X_test)
lg_acc = metrics.accuracy_score(y_test, lg_pred)
lg_auc = metrics.roc_auc_score(y_test, lg_pred)
lg_rep = metrics.classification_report(y_test, lg_pred)

'''Hyperparameter tuning - uncomment lines below to run'''

#solvers = ['newton-cg', 'lbfgs', 'liblinear']
#penalty = ['l2', 'l1']
#c_values = [1000, 500, 100, 50, 10, 5, 1.0, 0.1, 0.01, 0.001]

#lg_grid = dict(solver=solvers,penalty=penalty,C=c_values)
#lg_grid_search = GridSearchCV(estimator=lg, param_grid=lg_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
#lg_grid_result = lg_grid_search.fit(X_test, y_test)

#print("Best: %f using %s" % (lg_grid_result.best_score_, lg_grid_result.best_params_))


#%%
svm = SVC()

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = metrics.accuracy_score(y_test, svm_pred)
svm_auc = metrics.roc_auc_score(y_test, svm_pred)
svm_rep = metrics.classification_report(y_test, svm_pred)

'''Hyperparameter tuning - uncomment lines below to run'''

#kernel_values = ['rbf', 'poly', 'sigmoid']
#gammas = [0.001, 0.01, 0.1, 1]
#c_values =  [1000, 500, 100, 50, 10, 5, 1.0, 0.1, 0.01, 0.001]

#svm_grid = dict(kernel = kernel_values, gamma = gammas, C = c_values)
#svm_grid_search = GridSearchCV(estimator=svm, param_grid=svm_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
#svm_grid_result = svm_grid_search.fit(X_test, y_test)

#print("Best: %f using %s" % (svm_grid_result.best_score_, svm_grid_result.best_params_))

#%%

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = metrics.accuracy_score(y_test, rf_pred)
rf_auc = metrics.roc_auc_score(y_test, rf_pred)
rf_rep = metrics.classification_report(y_test, rf_pred)

'''WARNING - THIS ONE TAKES AN ETERNITY'''

#rf_grid = {
    #'max_depth': [80, 90, 100, 110],
    #'max_features': [2, 3],
    #'min_samples_leaf': [3, 4, 5],
    #'min_samples_split': [8, 10, 12],
    #'n_estimators': [100, 200, 300, 1000]}

#rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
#rf_grid_result = rf_grid_search.fit(X_test, y_test)

#print("Best: %f using %" % (rf_grid_result.best_score_, rf_grid_result.best_params_))
      
#%%
mlp = MLPClassifier()

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = metrics.accuracy_score(y_test, mlp_pred)
mlp_auc = metrics.roc_auc_score(y_test, mlp_pred)
mlp_rep = metrics.classification_report(y_test, mlp_pred)

'''WARNING - THIS ONE TAKES AN ETERNITY'''

#mlp_grid = {
    #'hidden_layer_sizes': [(10,30,10),(20,)],
    #'activation': ['tanh', 'relu'],
    #'solver': ['sgd', 'adam'],
    #'alpha': [0.0001, 0.05],
    #'learning_rate': ['constant','adaptive']}

#mlp_grid_search = GridSearchCV(estimator=mlp, param_grid=mlp_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
#mlp_grid_result = mlp_grid_search.fit(X_test, y_test)

#print("Best: %f using %" % (mlp_grid_result.best_score_, mlp_grid_result.best_params_))

#%%#Leave-one_out method to see which features contribute 

acc_scores = []
for i in range(len(feat_matrix.T)):
    print(i)
    lg_i = LogisticRegression()
    X_train_i = np.delete(X_train, obj = i, axis = 1 )
    X_test_i = np.delete(X_test, obj = i, axis = 1 )
    lg_i.fit(X_train_i, y_train)
    lg_pred_i = lg_i.predict(X_test_i)
    lg_acc_i = metrics.accuracy_score(y_test, lg_pred_i)
    acc_scores.append(lg_acc_i)