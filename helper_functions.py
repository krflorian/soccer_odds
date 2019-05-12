# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:10:17 2019

@author: kr_fl

soccer helper functions
"""
import requests
import pandas as pd

# get API key
key = open('rapid_api_key.txt', 'r').read()
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


