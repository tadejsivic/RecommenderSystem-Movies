from Predictor import Predictor

from UserItemData import UserItemData
from MovieData import MovieData

import numpy as np
import pandas as pd

class SlopeOnePredictor(Predictor):
# To use the slope one method, we need a target user. Then we need to recommend some target items - which have not yet been rated
# We tried the naive approach - which would work, but only in normal programming languages which actually run fast.
# Fit would be called every now and then, maybe overnight.

# Now, use the deviation method.
# Check fri.ucilnica

    # Compute all deviations for all pairs of movies.
    def fit(self, X:UserItemData):
        pass

    # Use the deviations computed in fit, to predict movies for a target user
    def predict(self, user_id, n):
        pass



sop = SlopeOnePredictor()
sop.fit(UserItemData("data/user_ratedmovies.dat", min_ratings=1000))

