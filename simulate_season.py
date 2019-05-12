# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:49:54 2019

@author: kr_fl
"""

import numpy as np

# read and clean data
odds = pd.read_excel('odds.xlsx')
odds = odds.dropna()
odds['team_2'] = [x.lstrip() for x in odds['team_2']]
odds['team_1'] = [x.rstrip() for x in odds['team_1']]
odds = odds.sort_index(ascending=False)

# add gameday
odds_rounds = [list(np.repeat(i, 9)) for i in results['round'].unique()]
odds_rounds = [item for sublist in odds_rounds for item in sublist]

odds['round'] = odds_rounds

# clean odds
odds[1] = [float(i[0:4]) for i in odds[1]]
odds['x'] = [float(i[0:4]) for i in odds['x']]
odds[2] = [float(i[0:4]) for i in odds[2]]

# add id column
team_id = ['184', '167', '165', '192', '157', '169', '159', '163', '175', '174', '170', 
           '168', '173', '164', '161', '162', '160', '181']
teams_german = pd.DataFrame({'id': team_id, 'team': odds['team_1'].unique()})

odds = (odds.merge(teams_german, left_on='team_1', right_on='team')
            .drop('team', axis=1)
            .rename( columns={'id': 'team_1_id'})
            .merge(teams_german, left_on='team_2', right_on='team')
            .drop('team', axis=1)
            .rename( columns={'id': 'team_2_id'}))
print(len(odds))
odds.head()



