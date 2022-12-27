from Predictor import Predictor

from UserItemData import UserItemData
from MovieData import MovieData

import numpy as np
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
        for p1 in np.unique(np.array(self.frame["movieID"])):
            for p2 in np.unique(np.array(self.frame["movieID"])):
                if p1 == p2:
                    continue

                if (p2, p1) in self.similar_movies.keys() :
                    self.similar_movies[(p1, p2)] = self.similar_movies[(p2, p1)]
                else: 
                    self.similar_movies[(p1, p2)] = self.similarity(p1, p2)


    def predict(self, user_id, num):
        pass


    def similarity(self, p1, p2):
        if p1 == p2:
            return 0
        # p1 and p2 are essentially movie_id1 and movie_id2
        # sim is the similarity score calculated using adjusted cosine similarity
        mask = (self.frame["movieID"] == p1) | (self.frame["movieID"] == p2)
        temporary_grand = self.frame[mask]    
        
        # First we get only proper ratings, so users who rated both movies
        temp_frame = pd.DataFrame()
        for user_id in np.unique(np.array(temporary_grand["userID"])):
            users_rated_movies = temporary_grand.loc[(temporary_grand["userID"] == user_id) & ((temporary_grand["movieID"] == p1) | (temporary_grand["movieID"] == p2)), ]
            if len(users_rated_movies) == 2:
                temp_frame = pd.concat([temp_frame, users_rated_movies])
        
        if len(temp_frame)/2 < self.min_values: # Not enough user ratings
            return 0

        # Then, we calculate the similarity using the adjusted cosine similarity formula
        numerator = 0
        denom1 = 0
        denom2 = 0
        
        if len(temp_frame) == 0:
            return 0
        
        for user_id in np.unique(np.array(temp_frame["userID"])):
            users_ratings = np.array(temp_frame.loc[temp_frame["userID"] == user_id, "rating"])
            users_average =  np.average(self.frame.loc[self.frame["userID"] == user_id, "rating"])
            
            numerator += (users_ratings[0] - users_average) * (users_ratings[1] - users_average)
            denom1 += math.pow(users_ratings[0] - users_average, 2)
            denom2 += math.pow(users_ratings[1] - users_average, 2)
        denom1 = math.sqrt(denom1)
        denom2 = math.sqrt(denom2)

        sim = numerator / (denom1 * denom2)
        
        if sim < self.threshold: # Insignificant similarity
            return 0
        return sim
    

ibp = ItemBasedPredictor()
ibp.fit(UserItemData("data/user_ratedmovies.dat", min_ratings=1000))

if (1580,780) in ibp.similar_movies.keys():
    print("1580, 780 is")
else:
    print("1580, 780 is nowhere to be found")
if (780,1580) in ibp.similar_movies.keys():
    print("780, 1580 is")
else:
    print("780, 1580 is nowhere to be found")

#print(ibp.similar_movies[(1,32)])
print(ibp.similar_movies[(1580, 780)])






