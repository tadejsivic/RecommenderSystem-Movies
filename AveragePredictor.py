from Predictor import Predictor
from UserItemData import UserItemData

import numpy as np
import pandas as pd

class AveragePredictor(Predictor):
    def __init__(self, b):
        self.b = b


    def fit(self, X:UserItemData):
        self.frame = pd.DataFrame()
        self.frame["userID"] = X.user_ratings["userID"]
        self.frame["movieID"] = X.user_ratings["movieID"]
        self.frame["rating"] = X.user_ratings["rating"]
        self.frame["rec_rating"] = -1

        g_avg = self.frame["rating"].sum() / len(self.frame["rating"])

        for id in np.unique(np.array(self.frame["movieID"])):
            mask = self.frame["movieID"] == id
            temp_frame = self.frame[mask]
            
            vs = temp_frame["rating"].sum()
            n = len(temp_frame)

            avg = (vs + self.b * g_avg) / (n + self.b)
            
            self.frame.loc[self.frame["movieID"] == id, "rec_rating"] = avg

#ap = AveragePredictor(0)
#ap.fit(UserItemData("data/user_ratedmovies.dat"))

