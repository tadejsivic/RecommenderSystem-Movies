from Predictor import Predictor

from UserItemData import UserItemData
from MovieData import MovieData

import numpy as np
import pandas as pd

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
            #self.frame.loc[self.frame["std"] not in max_stds, "rec_rating"] = -1
            if self.frame.iloc[i, 3] not in max_stds:
                self.frame.iloc[i, 4] = -1

#stdp = STDPredictor(100)
#stdp.fit(UserItemData("data/user_ratedmovies.dat"))
#stdp.predict(78, 5)
#print(stdp.frame)