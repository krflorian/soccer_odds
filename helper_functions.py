# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:10:17 2019

@author: kr_fl

soccer helper functions
"""
import requests
import pandas as pd

# get API key
key = open('data/rapid_api_key.txt', 'r').read()
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
def get_sample_row(gameday, last_gameday, team_1, team_2, results, i, table, stats, home_away):
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
        'form_pass_acc_1': sum(table[last_gameday][team_1]['form_passes'])/len(table[last_gameday][team_1]['form_passes']),   
        'form_shot_acc_1': sum(table[last_gameday][team_1]['form_accuracy'])/len(table[last_gameday][team_1]['form_accuracy']),
        'form_duels_1': sum(table[last_gameday][team_1]['form_duels'])/len(table[last_gameday][team_1]['form_duels']),
        'form_offense_1': sum(table[last_gameday][team_1]['form_offense'])/len(table[last_gameday][team_1]['form_offense']),
        'team_2': team_2,
        'goal_difference_2': table[last_gameday][team_2]['goal_difference'],
        'rating_2': table[last_gameday][team_2]['rating'],
        'form_2': sum(table[last_gameday][team_2]['form'])/len(table[last_gameday][team_2]['form']),
        'form_weighted_2' : form_weighted_2,
        'form_possession_2': sum(table[last_gameday][team_2]['form_possession'])/len(table[last_gameday][team_2]['form_possession']),
        'form_pass_acc_2': sum(table[last_gameday][team_2]['form_passes'])/len(table[last_gameday][team_2]['form_passes']),      
        'form_shot_acc_2': sum(table[last_gameday][team_2]['form_accuracy'])/len(table[last_gameday][team_2]['form_accuracy']),
        'form_duels_2': sum(table[last_gameday][team_2]['form_duels'])/len(table[last_gameday][team_2]['form_duels']),        
        'form_offense_2': sum(table[last_gameday][team_2]['form_offense'])/len(table[last_gameday][team_2]['form_offense']),
        'home_away': [home_away],
        'match_id': [results.loc[i]['match_id']],
        'target': [results.loc[i]['points_home']],
        'target_possession': stats[last_gameday][team_1]['Pässe (%)'][:2],
        'match_date': [results.loc[i]['match_date']],
        'B365H': [results.loc[i]['B365H']],
        'B365D': [results.loc[i]['B365D']],
        'B365A': [results.loc[i]['B365A']],})
    return sample

# get ensemble features
def get_ensemble_x(X, model_list):    
    i = 0
    for model in model_list:
        i += 1
        mdl_name = type(model).__name__
        y_hat = model.predict_proba(X)
        y_hat = pd.DataFrame(y_hat)
        y_hat.columns = [mdl_name+'0', mdl_name+'1', mdl_name+'3']
        if i == 1:
            X_ensemble = y_hat
        else:
            X_ensemble = X_ensemble.join(y_hat)
    return X_ensemble 

