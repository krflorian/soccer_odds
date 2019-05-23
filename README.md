# soccer_odds
Soccer fixtures prediction model. Uses ensemble of different machine learning models to increase prediction accuracy.


The Jupyter notebook simulation gives an overview of ensemble and prediction accuracy.

New Features to be added:
- Head to Head
- Pass Statistics
- Shot on Goal Statistics
- Players in team w/rating

API connection module:
- connects to api-football
- downloads historical fixtures for soccer leagues
- sets up features and training dataframe

Helper functions module:
- sets up python definitions

Train ensemble module:
- sets training pipeline for 5 different machine learning models
- model 1 - logistic regression (ridge)
- model 2 - knn
- model 3 - random forest
- model 4 - neural network
- ensemble - logistic regression (lasso)
- shows test set prediction accuracy


Simulation Module:
- Train Ensemble Models on last season
- Set Budget for betting
- Get Betting Odds for new season (not yet implemented)
- Setup training data for new season
- loop over gamedays (predict games,
					  bet according to threshold probability,
					  evaluate bet with historical odds,
					  train model with new data,
					  next iter)

Prediction Module: (not yet implemented)

