# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:08:46 2019

@author: kr_fl

Soccer API connection and data pipeline
"""
import os
os.chdir('E:\soccer')

import pandas as pd
import helper_functions as sc

# show leagues
leagues = sc.request_leagues()
for num in leagues:
    if leagues[num]['country'] == 'Germany':
        print(leagues[num])
# 2016 = 54, 2017 = 35

# load data
fixtures = sc.request_data(league=35, fixtures_teams='fixtures')
teams_info = sc.request_data(league=35, fixtures_teams='teams')

# get teams
teams = pd.DataFrame({'id': [], 'team': []})
for num in teams_info:
    teams = teams.append(pd.DataFrame({'id':[teams_info[str(num)]['team_id']], 'team': [teams_info[str(num)]['name']]}))
teams = teams.reset_index(drop=True)

# get results

results = pd.DataFrame()
for f in fixtures:
    results = results.append(pd.DataFrame({'round': [fixtures[f]['round']],
                                           'home': [fixtures[f]['homeTeam_id']],
                                           'away': [fixtures[f]['awayTeam_id']],
                                           'goals_home': [fixtures[f]['goalsHomeTeam']],
                                           'goals_away': [fixtures[f]['goalsAwayTeam']]}))
results = results.reset_index(drop=True)

# get results
results['points_home'] = results.apply(lambda x: sc.points(x['goals_home'], x['goals_away']), axis=1)
results['points_away'] = results.apply(lambda x: sc.points(x['goals_away'], x['goals_home']), axis=1)

# setup dictionary
rounds = results['round'].unique()
table = {r:{t:{'points':[], 'goals_shot':[], 'goals_received':[]} for t in teams['id']} for r in rounds}

# setup table
last_round = ''
for r in rounds:
    for t in teams['id']:
        try:
            points = int(results[(results['round'] == r) & (results['home'] == str(t))]['points_home'])
            goals_shot = int(results[(results['round'] == r) & (results['home'] == str(t))]['goals_home'])
            goals_received = int(results[(results['round'] == r) & (results['home'] == str(t))]['goals_away'])
        except:
            points = int(results[(results['round'] == r) & (results['away'] == str(t))]['points_away'])
            goals_shot = int(results[(results['round'] == r) & (results['away'] == str(t))]['goals_away'])
            goals_received = int(results[(results['round'] == r) & (results['away'] == str(t))]['goals_home'])
        if last_round == '':
            table[r][str(t)]['points'] = points
            table[r][str(t)]['points_cum'] = points
            table[r][str(t)]['goals_shot'] = goals_shot
            table[r][str(t)]['goals_received'] = goals_received
        else:
            table[r][str(t)]['points'] = points
            table[r][str(t)]['points_cum'] = table[last_round][str(t)]['points_cum'] + points
            table[r][str(t)]['goals_shot'] = table[last_round][str(t)]['goals_shot'] + goals_shot
            table[r][str(t)]['goals_received'] = table[last_round][str(t)]['goals_received'] + goals_received
        table[r][str(t)]['goal_difference'] = table[r][str(t)]['goals_shot']-table[r][str(t)]['goals_received']
    last_round = r
#table
    
# get rank & rating
for r in rounds:
    gameday = pd.DataFrame()
    for t in teams['id']:
        gameday = gameday.append(pd.DataFrame({'round': [r],
                                               'team': [t],
                                               'points_cum': table[r][str(t)]['points_cum'],
                                               'goal_difference': table[r][str(t)]['goal_difference']}))
    gameday = gameday.sort_values(['points_cum', 'goal_difference'], ascending = False)
    gameday['rank'] = [i for i in range(1,len(gameday)+1)]
    for t in teams['id']:
        table[r][str(t)]['rank'] = int(gameday[gameday['team'] == t]['rank'])
        table[r][str(t)]['rating'] = sc.get_rating(int(table[r][str(t)]['rank']))
        

# get form
for t in teams['id']:
    form = []
    form_rating = []
    for r in rounds:
        if len(form) > 4:
            form.pop(0)
            form_rating.pop(0)
        form.append(table[r][t]['points'])
        form_rating.append(table[r][t]['rating'])
        table[r][t]['form'] = form.copy()
        table[r][t]['form_rating'] = form_rating.copy()
        

# setup training data
data = pd.DataFrame()
for i in range(len(results)):
    team_1 = results.loc[i]['home']
    team_2 = results.loc[i]['away']
    gameday = results.loc[i]['round']
    if gameday == 'Bundesliga - 1':
        last_gameday = gameday
        next
    form_ratings_1 = sc.get_rating_value(table[last_gameday][team_1]['form_rating'])
    form_ratings_2 = sc.get_rating_value(table[last_gameday][team_2]['form_rating'])
    form_weighted_1 = sum([table[last_gameday][team_1]['form'][i]*form_ratings_1[i] for i in range(len(form_ratings_1))])
    form_weighted_2 = sum([table[last_gameday][team_2]['form'][i]*form_ratings_2[i] for i in range(len(form_ratings_2))])
    if form_weighted_1 > 0:
        form_weighted_1 = form_weighted_1/len(form_ratings_1)
    if form_weighted_2 > 0:
        form_weighted_2 = form_weighted_1/len(form_ratings_2)
    data = data.append(pd.DataFrame({
                                    'round': gameday,
                                    'team_1': team_1,
                                    'goal_difference_1': table[last_gameday][team_1]['goal_difference'],
                                    'rating_1': table[last_gameday][team_1]['rating'],
                                    'form_1': sum(table[last_gameday][team_1]['form'])/len(table[last_gameday][team_1]['form']),
                                    'form_weighted_1': form_weighted_1,                            
                                    'team_2': team_2,
                                    'goal_difference_2': table[last_gameday][team_2]['goal_difference'],
                                    'rating_2': table[last_gameday][team_2]['rating'],
                                    'form_2': sum(table[last_gameday][team_2]['form'])/len(table[last_gameday][team_2]['form']),
                                    'form_weighted_2' : form_weighted_2,
                                    'home_away': [1],
                                    'target': [results.loc[i]['points_home']]
                                    }))
    data = data.append(pd.DataFrame({
                                'round': gameday,
                                'team_1': team_2,
                                'goal_difference_1': table[last_gameday][team_2]['goal_difference'],
                                'rating_1': table[last_gameday][team_2]['rating'],
                                'form_1': sum(table[last_gameday][team_2]['form'])/len(table[last_gameday][team_2]['form']),
                                'form_weighted_1' : form_weighted_2,                        
                                'team_2': team_1,
                                'goal_difference_2': table[last_gameday][team_1]['goal_difference'],
                                'rating_2': table[last_gameday][team_1]['rating'],    
                                'form_2': sum(table[last_gameday][team_1]['form'])/len(table[last_gameday][team_1]['form']),
                                'form_weighted_2': form_weighted_1,
                                'home_away': [0],
                                'target': [results.loc[i]['points_away']]
                                }))
    last_gameday = gameday
data = data.reset_index(drop=True)

# get dummy columns
team_1 = pd.get_dummies(data['team_1'], prefix = 'team_1_', drop_first = True)
team_2 = pd.get_dummies(data['team_2'], prefix = 'team_2_', drop_first = True)
rating_1 = pd.get_dummies(data['rating_1'], prefix = 'team_1', drop_first = True)
rating_2 = pd.get_dummies(data['rating_2'], prefix = 'team_2', drop_first = True)

# join tables to get training data
data_ready = (data.join(rating_1)
                  .join(rating_2)
                  .drop(['team_1', 'team_2', 'rating_1', 'rating_2', 'round'], axis=1))

# save csv files
data_ready.to_csv('data\\bundesliga_2016_ready.csv')
data_ready.to_csv('data\\bundesliga_2016_full.csv')

f = open("data\\bundesliga_2016_dict.txt","w")
f.write( str(table) )
f.close()

