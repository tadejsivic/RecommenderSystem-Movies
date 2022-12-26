from Predictor import Predictor
from UserItemData import UserItemData
from MovieData import MovieData

import numpy as np
import pandas as pd

class ViewsPredictor(Predictor):
    # Constructor not needed. Not even "predict". This class simply counts the number of views
    def fit(self, X:UserItemData):
        self.frame = pd.DataFrame()
        self.frame["userID"] = X.user_ratings["userID"]
        self.frame["movieID"] = X.user_ratings["movieID"]
        self.frame["rating"] = X.user_ratings["rating"]
        # we leave the column name as "rec_rating", because of Recommender - NViews or numOfViews would of course be better
        self.frame["rec_rating"] = 0

        for id in np.unique(np.array(self.frame["movieID"])):
            mask = self.frame["movieID"] == id
            temp_frame = self.frame[mask]
            
            self.frame.loc[self.frame["movieID"] == id, "rec_rating"] = len(temp_frame)

#p = ViewsPredictor()
#p.fit(UserItemData("data/user_ratedmovies.dat"))
#rint(p.frame)

