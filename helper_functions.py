# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:10:17 2019

@author: kr_fl

soccer helper functions
"""
import requests
import pandas as pd

# get API key
key = open('data\\rapid_api_key.txt', 'r').read()
# create functions
# get results
def request_data(league='', fixtures_teams=''):
    request = requests.get("https://api-football-v1.p.rapidapi.com/" + fixtures_teams + "/league/" + str(league) ,
           headers={
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
            "X-RapidAPI-Key": key
           })
    data = pd.read_json(request.content).loc[fixtures_teams][0]
    return data

# get list of leagues
def request_leagues():
    request = requests.get("https://api-football-v1.p.rapidapi.com/leagues/",
           headers={
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
            "X-RapidAPI-Key": key
           })
    data = pd.read_json(request.content).loc['leagues',][0]
    return data

# get statistics
def request_match_statistics(match_id):
    request = requests.get("https://api-football-v1.p.rapidapi.com/v2/statistics/fixture/" + str(match_id)+  "/",
           headers={
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
            "X-RapidAPI-Key": key
           })
    data = pd.read_json(request.content).loc['statistics',][0]
    return data
	
# get points
def points(home, away):
    if home > away:
        return 3
    elif home < away:
        return 0
    else: return 1

# get rating
def get_rating(rank):
    if rank <= 3:
        return 'A'
    elif rank <= 6:
        return 'B'
    elif rank <= 15:
        return 'C'
    else:
        return 'D'

# get rating values
def get_rating_value(rating):
    rating_list = []
    for i in rating:
        if i == 'A':
            rating_list.append(4)
        elif i == 'B':
            rating_list.append(3)
        elif i == 'C':
            rating_list.append(2)
        else:
            rating_list.append(1)
    return rating_list
	
# get sample with features and target
def get_sample_row(gameday, last_gameday, team_1, team_2, results, i, table):
    form_ratings_1 = get_rating_value(table[last_gameday][team_1]['form_rating'])
    form_ratings_2 = get_rating_value(table[last_gameday][team_2]['form_rating'])
    form_weighted_1 = sum([table[last_gameday][team_1]['form'][i]*form_ratings_1[i] for i in range(len(form_ratings_1))])
    form_weighted_2 = sum([table[last_gameday][team_2]['form'][i]*form_ratings_2[i] for i in range(len(form_ratings_2))])
    if form_weighted_1 > 0:
        form_weighted_1 = form_weighted_1/len(form_ratings_1)
    if form_weighted_2 > 0:
        form_weighted_2 = form_weighted_1/len(form_ratings_2)
    sample = pd.DataFrame({
        'round': gameday,
        'team_1': team_1,
        'goal_difference_1': table[last_gameday][team_1]['goal_difference'],
        'rating_1': table[last_gameday][team_1]['rating'],
        'form_1': sum(table[last_gameday][team_1]['form'])/len(table[last_gameday][team_1]['form']),
        'form_weighted_1': form_weighted_1, 
        'form_possession_1': sum(table[last_gameday][team_1]['form_possession'])/len(table[last_gameday][team_1]['form_possession']),
        'form_pass_acc_1': sum(table[last_gameday][team_1]['form_pass_acc'])/len(table[last_gameday][team_1]['form_pass_acc']),   
        'form_shot_acc_1': sum(table[last_gameday][team_1]['form_shot_acc'])/len(table[last_gameday][team_1]['form_shot_acc']),
        'team_2': team_2,
        'goal_difference_2': table[last_gameday][team_2]['goal_difference'],
        'rating_2': table[last_gameday][team_2]['rating'],
        'form_2': sum(table[last_gameday][team_2]['form'])/len(table[last_gameday][team_2]['form']),
        'form_weighted_2' : form_weighted_2,
        'form_possession_2': sum(table[last_gameday][team_2]['form_possession'])/len(table[last_gameday][team_2]['form_possession']),
        'form_pass_acc_2': sum(table[last_gameday][team_2]['form_pass_acc'])/len(table[last_gameday][team_2]['form_pass_acc']),      
        'form_shot_acc_2': sum(table[last_gameday][team_2]['form_shot_acc'])/len(table[last_gameday][team_2]['form_shot_acc']),
        'home_away': [1],
        'target': [results.loc[i]['points_home']] })
    return sample	

# get ensemble features
def get_ensemble_x(X, mdl_ridge, mdl_knn, mdl_rf, mdl_mlp):
        # ensemble
    y_hat_ridge = mdl_ridge.predict_proba(X)
    y_hat_knn = mdl_knn.predict_proba(X)
    y_hat_rf = mdl_rf.predict_proba(X)
    y_hat_mlp = mdl_mlp.predict_proba(X)

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
    
    return X_ensemble