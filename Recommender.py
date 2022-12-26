from RandomPredictor import RandomPredictor
from AveragePredictor import AveragePredictor
from Predictor import Predictor
from ViewsPredictor import ViewsPredictor
from STDPredictor import STDPredictor

from UserItemData import UserItemData
from MovieData import MovieData

import pandas as pd
import numpy as np

class Recommender:
    def __init__(self, predictor:Predictor):
        self.predictor = predictor


    def fit(self, X):
        self.predictor.fit(X)
        self.frame = self.predictor.frame


    def recommend(self, user_id, n=10, rec_seen=True):
        self.predictor.predict(user_id, n)
        print("Started sorting")
        self.frame.sort_values("rec_rating", ascending=False, inplace=True)
        if not rec_seen:
            mask = self.frame["userID"] != user_id
            self.frame = self.frame[mask]
        result = []
        # TO OPTIMIZIRAJ!!!!!!! python je one of the languages ever made
        i=0
        while len(result) < n and i < len(self.frame):
            movie_id, rec_rating = (self.frame[["movieID", "rec_rating"]].iloc[i])
            if (movie_id, rec_rating) not in result:
                result.append((movie_id,rec_rating))
            i += 1
        return result


md = MovieData('data/movies.dat')
uim = UserItemData('data/user_ratedmovies.dat')
rec = Recommender(STDPredictor(100))
rec.fit(uim)
rec_items = rec.recommend(78, n=5, rec_seen=False)
for idmovie, val in rec_items:
    try:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
    except Exception:
        print("Movie does not have any ratings")
