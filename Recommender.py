from RandomPredictor import RandomPredictor
from AveragePredictor import AveragePredictor
from Predictor import Predictor
from ViewsPredictor import ViewsPredictor
from STDPredictor import STDPredictor
from ItemBasedPredictor import ItemBasedPredictor

from UserItemData import UserItemData
from MovieData import MovieData

import pandas as pd
import numpy as np

class Recommender:
    def __init__(self, predictor:Predictor):
        self.predictor = predictor


    def fit(self, X):
        self.predictor.fit(X)

    def recommend(self, user_id, n=10, rec_seen=True):
        predicted_frame = self.predictor.predict(user_id, n)
        predicted_frame.sort_values("rec_rating", ascending=False, inplace=True)
        if not rec_seen:
            mask = predicted_frame["userID"] != user_id
            predicted_frame = predicted_frame[mask]
        result = []
        i=0
        while len(result) < n and i < len(predicted_frame):
            movie_id, rec_rating = (predicted_frame[["movieID", "rec_rating"]].iloc[i])
            if (movie_id, rec_rating) not in result:
                result.append((movie_id,rec_rating))
            i += 1
        return result


    def recommend_similar_items(self, movie_id, num):
        predicted_frame = self.predictor.similarItems(movie_id, num)
        predicted_frame.sort_values("sim", ascending=False, inplace=True)
        result = []
        for i in range(len(predicted_frame)):
            movie_id, sim = (predicted_frame[["movieID", "sim"]].iloc[i])
            if (movie_id, sim) not in result:
                result.append((movie_id, sim))
        return result



md = MovieData('data/movies.dat')
uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
pred = ItemBasedPredictor()
rec = Recommender(pred)
rec.fit(uim)
#rec_items = rec.recommend(1, n=10, rec_seen=True)
#for idmovie, val in rec_items:
#    try:
#        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
#    except Exception:
#        print("Movie does not have any ratings")
#pred.print_most_similar_movies(md, 20)

rec_items = rec.recommend_similar_items(4993, 10)
print('Filmi podobni "The Lord of the Rings: The Fellowship of the Ring": ')
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))


