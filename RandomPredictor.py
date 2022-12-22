import pandas as pd
import numpy as np
import random

from UserItemData import UserItemData
from MovieData import MovieData

class RandomPredictor:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def fit(self, X : UserItemData):
        self.frame = pd.DataFrame()
        self.frame["userID"] = X.user_ratings["userID"]
        self.frame["movieID"] = X.user_ratings["movieID"]
        self.frame["rec_rating"] = np.random.randint(self.min, self.max + 1, size=len(self.frame))
        #print(self.frame)


    def predict(self, user_id):
        # The user_id is not needed here, since we randomize the same for everyone
        movie_rating = {}
        for movie_id, rec_rating in zip(self.frame["movieID"], self.frame["rec_rating"]):
            movie_rating[movie_id] = rec_rating
        return movie_rating


#md = MovieData('data/movies.dat')
#uim = UserItemData('data/user_ratedmovies.dat')
#rp = RandomPredictor(1, 5)
#rp.fit(uim)
#pred = rp.predict(78)
#print(type(pred))
#items = [1, 3, 20, 50, 100]
#for item in items:
#   print("Film: {}, ocena: {}".format(md.get_title(item), pred[item]))