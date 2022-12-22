from RandomPredictor import RandomPredictor
from UserItemData import UserItemData
from MovieData import MovieData

import pandas as pd
import numpy as np

class Recommender:
    def __init__(self, predictor:RandomPredictor):
        self.predictor = predictor
        

    def fit(self, X):
        self.predictor.fit(X)
        self.frame = self.predictor.frame

    def recommend(self, user_id, n=10, rec_seen=True):
        self.frame.sort_values("rec_rating", ascending=False, inplace=True)
        if rec_seen:
            mask = self.frame["userID"] != user_id
            self.frame = self.frame[mask]
        result = []
        for i in range(n):
            result.append( zip(self.frame[["movieID", "rec_rating"]].iloc[i]))
        return result


md = MovieData('data/movies.dat')
uim = UserItemData('data/user_ratedmovies.dat')
rp = RandomPredictor(1, 5)
rec = Recommender(rp)
rec.fit(uim)
rec_items = rec.recommend(78, n=5, rec_seen=True)
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))

