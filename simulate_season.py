# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:49:54 2019

@author: kr_fl
"""
import os 
os.chdir('E:\soccer')

import pandas as pd
import numpy as np
import helper_functions as sc

from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


data_ready = pd.read_csv('data\\bundesliga_2016_ready.csv', index_col=0)

X_train = data_ready.drop(['target'], axis= 1)
y_train = data_ready['target']
features = list(X_train.columns)

# Normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_train.columns = features


# train inicial models
mdl_ridge = LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=100, multi_class = 'auto') # Ridge
mdl_ridge.fit(X_train,y_train)

mdl_knn = KNeighborsClassifier(n_neighbors=29)  
mdl_knn.fit(X_train,y_train)

mdl_rf = RandomForestClassifier(min_samples_leaf=5, random_state=1, n_jobs=-1,
                                n_estimators=1000, max_depth=5)
mdl_rf.fit(X_train,y_train)

mdl_mlp = MLPClassifier(hidden_layer_sizes = (10, ), alpha = 10,
                        activation='relu', solver='lbfgs',
                        batch_size=20, learning_rate='invscaling', learning_rate_init=0.001)
mdl_mlp.fit(X_train,y_train)

print('finished training inicial models')

# ensemble
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

# train ensemble model
mdl_lasso = LogisticRegression(C = 10, penalty='l1', solver='liblinear',
                               max_iter=100, multi_class = 'auto')
mdl_lasso.fit(X_ensemble, y_train)

print('finished training all models')

##############
# Simulation #
##############

data_new = pd.read_csv('data\\bundesliga_2017_ready.csv', index_col=0)
results = pd.DataFrame()

print('starting simulation')

for gameday in data_new['round'].unique():
    gameday_data = data_new[data_new['round'] == gameday]
    
    X = gameday_data.drop(['target', 'round'], axis= 1)
    y = gameday_data['target']
    features = list(X.columns)
    
    # Normalization
    X = scaler.transform(X)
    X = pd.DataFrame(X)
    X.columns = features
    
    X_ensemble = sc.get_ensemble_x(X, mdl_ridge, mdl_knn, mdl_rf, mdl_mlp)
    
    y_hat = mdl_lasso.predict(X_ensemble)
    y_hat_proba = mdl_lasso.predict_proba(X_ensemble)
    bet = [any(y_hat_proba[i] > 0.8) for i in range(len(y_hat_proba))]

    results = results.append(pd.DataFrame({'gameday': list(np.repeat(gameday, len(y))),
                                           'y':y,
                                           'y_hat':y_hat,
                                           'bet': bet}))
    
    # retrain models
    X_train = X_train.append(X)
    y_train = y_train.append(y)
    
    mdl_ridge.fit(X_train, y_train)
    mdl_knn.fit(X_train, y_train)
    mdl_rf.fit(X_train, y_train)
    mdl_mlp.fit(X_train, y_train)
    
    X_ensemble = sc.get_ensemble_x(X_train, mdl_ridge, mdl_knn, mdl_rf, mdl_mlp)
    
    mdl_lasso.fit(X_ensemble, y_train)
    
    print('finished simulating gameday: ', gameday, '\n')
    

print('final accuracy of ensemble on test set = \n', accuracy_score(results['y'], results['y_hat']))
print(confusion_matrix(results['y'], results['y_hat']))  
print(classification_report(results['y'], results['y_hat'])) 

results_bet = results[results['bet:'] == True]
print('final accuracy of ensemble on test set with threshold = \n', accuracy_score(results_bet['y'], results_bet['y_hat']))
print('number of bets = ', len(results_bet))

# final accuracy of ensemble on test set BL 2017 without retraining = 0.5686274509803921
# final accuracy of ensemble on test set BL 2017 with retraining = 0.6029411764705882
# final accuracy of ensemble on test set BL 2017 with retreaining and threshold = 0.6734104046242775

#########################################################################################
# read and clean data
#odds = pd.read_excel('odds.xlsx')
#odds = odds.dropna()
#odds['team_2'] = [x.lstrip() for x in odds['team_2']]
#odds['team_1'] = [x.rstrip() for x in odds['team_1']]
#odds = odds.sort_index(ascending=False)
#
## add gameday
#odds_rounds = [list(np.repeat(i, 9)) for i in results['round'].unique()]
#odds_rounds = [item for sublist in odds_rounds for item in sublist]
#
#odds['round'] = odds_rounds
#
## clean odds
#odds[1] = [float(i[0:4]) for i in odds[1]]
#odds['x'] = [float(i[0:4]) for i in odds['x']]
#odds[2] = [float(i[0:4]) for i in odds[2]]
#
## add id column
#team_id = ['184', '167', '165', '192', '157', '169', '159', '163', '175', '174', '170', 
#           '168', '173', '164', '161', '162', '160', '181']
#teams_german = pd.DataFrame({'id': team_id, 'team': odds['team_1'].unique()})
#
#odds = (odds.merge(teams_german, left_on='team_1', right_on='team')
#            .drop('team', axis=1)
#            .rename( columns={'id': 'team_1_id'})
#            .merge(teams_german, left_on='team_2', right_on='team')
#            .drop('team', axis=1)
#            .rename( columns={'id': 'team_2_id'}))
#print(len(odds))
#odds.head()



