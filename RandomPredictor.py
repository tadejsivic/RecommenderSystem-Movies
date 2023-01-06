import pandas as pd
import numpy as np

from UserItemData import UserItemData
from MovieData import MovieData
from Predictor import Predictor

# Random predictor da vsem filmom naključno
# Uporabi se ga v recommenderju


class RandomPredictor(Predictor):
    def __init__(self, min, max):
        self.min = min
        self.max = max


    def fit(self, X : UserItemData):
        self.frame = pd.DataFrame()
        self.frame["userID"] = X.user_ratings["userID"]
        self.frame["movieID"] = X.user_ratings["movieID"]
        self.frame["rec_rating"] = np.random.randint(self.min, self.max + 1, size=len(self.frame))


    def predict(self, user_id, n=-1):
        # The user_id is not needed here, since we randomize the same for everyone
        self.frame["userID"] = user_id
        return self.frame


