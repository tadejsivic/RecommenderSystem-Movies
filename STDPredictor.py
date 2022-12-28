from Predictor import Predictor

from UserItemData import UserItemData
from MovieData import MovieData

import numpy as np
import pandas as pd

# This thing takes way to long, but I wanted to make it dynamic
# So it calculates the STD - Standard deviation of rating for each movie, then it sorts by the highest STD.
# When called, it returns N movies with highest STD
# Why does it take so long? Because we're using Python's core functionalities

class STDPredictor(Predictor):
    def __init__(self, n):
        self.n = n

    def fit(self, X:UserItemData):
        self.frame = pd.DataFrame()
        self.frame["userID"] = X.user_ratings["userID"]
        self.frame["movieID"] = X.user_ratings["movieID"]
        self.frame["rating"] = X.user_ratings["rating"]
        self.frame["std"] = -1
        self.frame["rec_rating"] = -1

        for id in np.unique(np.array(self.frame["movieID"])):
            mask = self.frame["movieID"] == id
            temp_frame = self.frame[mask]
            
            if len(temp_frame) < self.n:
                continue
            
            std = np.std(temp_frame["rating"])
            self.frame.loc[self.frame["movieID"] == id, "std"] = std
            self.frame.loc[self.frame["movieID"] == id, "rec_rating"] = np.average(temp_frame["rating"])

    def predict(self, user_id, num):
        max_stds = np.sort(np.unique(np.array(self.frame["std"])))[-num:]
        for i in range(len(self.frame)):
            # If STD is not one of the N highest, set rec_rating to -1, that way it will ignore those
            if self.frame.iloc[i, 3] not in max_stds:
                self.frame.iloc[i, 4] = -1
        return self.frame

#stdp = STDPredictor(100)
#stdp.fit(UserItemData("data/user_ratedmovies.dat"))
#stdp.predict(78, 5)
#print(stdp.frame)