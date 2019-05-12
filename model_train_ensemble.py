# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:37:21 2019

@author: kr_fl

Ensemble Machine Learning
"""
import os
os.chdir('E:\soccer')

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


X = data_ready.drop(['target'], axis= 1)
y = data_ready['target']

## get train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1 , test_size=0.2)



#########################################
# Model 1 - logistic regression - Ridge #
#########################################

mdl_ridge = LogisticRegression(penalty='l2', solver='liblinear', max_iter=100, multi_class = 'auto') # Ridge

# folds
kfold = KFold(n_splits=5, shuffle = True, random_state = 1)

# gridsearch cv
param_grid = [{
    'C': [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
}]

search = GridSearchCV(mdl_ridge, param_grid, cv=kfold, 
                      scoring = 'accuracy',
                      return_train_score=True)
search.fit(X_train, y_train)
mdl_ridge = search.best_estimator_

print(mdl_ridge)
print(search.best_score_)


##################################
# Model 2 - k-neirest neighbors ##
##################################

mdl_knn = KNeighborsClassifier()  

# folds
kfold = KFold(n_splits=5, shuffle = True, random_state = 1)

# gridsearch cv
param_grid = [{
    'n_neighbors': [i for i in range(1,100)]
}]

search = GridSearchCV(mdl_knn, param_grid, cv=kfold, 
                      scoring = 'accuracy',
                      return_train_score=True)

search.fit(X_train, y_train)
mdl_knn = search.best_estimator_

print(mdl_knn)
print(search.best_score_)

###########################
# model 3 - random forest #
###########################

mdl_rf = RandomForestClassifier(min_samples_leaf=5, random_state=1, n_jobs=-1)

# folds
kfold = KFold(n_splits=5, shuffle = True, random_state = 1)

# gridsearch cv
param_grid = [{
    'n_estimators': [500, 1000],
    'max_depth': [i for i in range(5,15)],
    'max_features': ['sqrt', 'log2']
}]

search = GridSearchCV(mdl_rf, param_grid, cv=kfold, 
                      scoring = 'accuracy',
                      return_train_score=True)

search.fit(X_train, y_train)
mdl_rf = search.best_estimator_

print(mdl_rf)
print(search.best_score_)

feature_importance = (pd.DataFrame({'features': X_train.columns,
                                    'importance': mdl_rf.feature_importances_})
                        .sort_values('importance', ascending=False))
print(feature_importance)

#############################
# model 4 - neural network  #
#############################

mdl_mlp = MLPClassifier(hidden_layer_sizes = (50, 50), activation='relu', alpha=0.0001, solver='lbfgs',
                        batch_size=20, learning_rate='invscaling', learning_rate_init=0.001)

kfold = KFold(n_splits=5, shuffle = True, random_state = 1)

# gridsearch cv
param_grid = [{
    'hidden_layer_sizes': [(10,), (50,), (100,), (150,)],
    'alpha': [0.0001, 0.001, 0.01, 1, 10]
}]

search = GridSearchCV(mdl_mlp, param_grid, cv=kfold, 
                      scoring = 'accuracy',
                      return_train_score=True)

search.fit(X_train, y_train)
mdl_mlp = search.best_estimator_

print(mdl_mlp)
print(search.best_score_)

#################################################
# Ensemble - setup logistic regression - lasso  #
#################################################

# predict all models
y_hat_ridge = mdl_ridge.predict_proba(X_train)
y_hat_knn = mdl_knn.predict_proba(X_train)
y_hat_rf = mdl_rf.predict_proba(X_train)
y_hat_mlp = mdl_mlp.predict_proba(X_train)

# create dataframes with output probability values
y_hat_rf = pd.DataFrame(y_hat_rf)
y_hat_rf.columns = ['rf_0', 'rf_1', 'rf_3']
y_hat_knn = pd.DataFrame(y_hat_knn)
y_hat_knn.columns = ['knn_0', 'knn_1', 'knn_3']
y_hat_ridge = pd.DataFrame(y_hat_ridge)
y_hat_ridge.columns = ['ridge_0', 'ridge_1', 'ridge_3']
y_hat_mlp = pd.DataFrame(y_hat_mlp)
y_hat_mlp.columns = ['mlp_0', 'mlp_1', 'mlp_3']

X_ensemble = y_hat_rf.join(y_hat_ridge).join(y_hat_knn).join(y_hat_mlp)
X_ensemble.head()


# train logistic regression ensemble
mdl_lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100, multi_class = 'auto')

# folds
kfold = KFold(n_splits=5, shuffle = True, random_state = 1)

# gridsearch cv
param_grid = [{
    'C': [3000, 2000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
}]

search = GridSearchCV(mdl_lasso, param_grid, cv=kfold, 
                      scoring = 'accuracy',
                      return_train_score=True)

search.fit(X_ensemble, y_train.reset_index(drop=True))
mdl_ensemble = search.best_estimator_

print(mdl_ensemble)
print(search.best_score_)

###################################################################################
# save models




##################################################################################



############################
# TEST model performance  ##
############################

# rename!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
y_hat_ridge = mdl_ridge.predict_proba(X_test)
y_hat_knn = mdl_knn.predict_proba(X_test)
y_hat_rf = mdl_rf.predict_proba(X_test)
y_hat_mlp = mdl_mlp.predict_proba(X_test)

y_hat_rf = pd.DataFrame(y_hat_rf)
y_hat_rf.columns = ['rf_0', 'rf_1', 'rf_3']
y_hat_knn = pd.DataFrame(y_hat_knn)
y_hat_knn.columns = ['knn_0', 'knn_1', 'knn_3']
y_hat_ridge = pd.DataFrame(y_hat_ridge)
y_hat_ridge.columns = ['ridge_0', 'ridge_1', 'ridge_3']
y_hat_mlp = pd.DataFrame(y_hat_mlp)
y_hat_mlp.columns = ['mlp_0', 'mlp_1', 'mlp_3']

X_ensemble = y_hat_rf.join(y_hat_ridge).join(y_hat_knn).join(y_hat_mlp)

coefficients = pd.DataFrame({'coefficients: ': list(X_ensemble.columns),
                             'win': mdl_ensemble.coef_[0],
                             'draw':mdl_ensemble.coef_[1],
                             'loose': mdl_ensemble.coef_[2]})

y_hat_ensemble = mdl_ensemble.predict(X_ensemble)

print('final accuracy of ensemble on test set = \n', accuracy_score(y_test, y_hat_ensemble))
print(confusion_matrix(y_test, y_hat_ensemble))  
print(classification_report(y_test, y_hat_ensemble)) 


###############################
# setup threshold  
###############################

y_hat_proba = mdl_ensemble.predict_proba(X_ensemble)
bet = [any(y_hat_proba[i] > 0.7) for i in range(len(y_hat_proba))]

data_test = data.loc[X_test.index,]
data_test['predict'] = y_hat_ensemble
data_test['bet'] = bet

print('number of bets = ', len(data_test[bet]), '\n',
      'accuracy = ', accuracy_score(data_test[bet]['target'], data_test[bet]['predict']), '\n',
      'confusiont matrix = ', '\n','\n',
     confusion_matrix(data_test[bet]['target'], data_test[bet]['predict']))
data_test.head()



