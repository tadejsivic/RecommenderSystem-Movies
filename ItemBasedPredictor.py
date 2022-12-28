from Predictor import Predictor

from UserItemData import UserItemData
from MovieData import MovieData

import numpy as np
from numpy import linalg
import pandas as pd
import math

class ItemBasedPredictor(Predictor):
    def __init__(self, min_values = 0, threshold = 0):
        self.min_values = min_values
        self.threshold = threshold


    def fit(self, X:UserItemData):
        self.frame = pd.DataFrame()
        self.frame["userID"] = X.user_ratings["userID"]
        self.frame["movieID"] = X.user_ratings["movieID"]
        self.frame["rating"] = X.user_ratings["rating"]
        self.frame["rec_rating"] = 0

        self.similar_movies = {}
        unique_movies = self.frame["movieID"].unique()
        user_averages_data = self.frame.groupby(["userID"])["rating"].mean()
        user_averages = dict(zip( user_averages_data.index.tolist(), user_averages_data.values.tolist() ))

        for p1 in unique_movies:
            for p2 in unique_movies:
                if p1 == p2:
                    continue
                if (p1,p2) in self.similar_movies: 
                    self.similar_movies[(p2,p1)] = self.similar_movies[(p1,p2)]
                    continue
                elif (p2,p1) in self.similar_movies:
                    self.similar_movies[(p1,p2)] = self.similar_movies[(p2,p1)]
                    continue
                
                # Similarity
                first_movie = self.frame[self.frame["movieID"] == p1]
                second_movie = self.frame[self.frame["movieID"] == p2]
                users_both_movies = np.intersect1d( first_movie["userID"].unique(), second_movie["userID"].unique() )
                if len(users_both_movies) < self.min_values:
                    self.similar_movies[(p1,p2)] = 0
                    continue

                first_movie = first_movie.loc[first_movie["userID"].isin(users_both_movies)]
                second_movie = second_movie.loc[second_movie["userID"].isin(users_both_movies)]
                
                user_averages_ratings = [user_averages[x] for x in users_both_movies]
                first_movie_ratings = np.array(first_movie["rating"].tolist()) - user_averages_ratings
                second_movie_ratings = np.array(second_movie["rating"].tolist()) - user_averages_ratings
                
                # We calculate the similarity using the adjusted cosine similarity formula                
                sim = np.dot(first_movie_ratings, second_movie_ratings) / (np.linalg.norm(first_movie_ratings) * np.linalg.norm(second_movie_ratings))
                if sim < self.threshold: # Insignificant similarity
                    self.similar_movies[(p1,p2)] = 0
                    continue
                self.similar_movies[(p1,p2)] = sim
                

    def predict(self, user_id, num):
        pass


    def similarity(self, p1, p2):
        return self.similar_movies[(p1,p2)]


ibp = ItemBasedPredictor()
ibp.fit(UserItemData("data/user_ratedmovies.dat", min_ratings=1000))

print(ibp.similarity(1580, 780))






