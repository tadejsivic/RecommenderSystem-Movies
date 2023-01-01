from RandomPredictor import RandomPredictor
from AveragePredictor import AveragePredictor
from Predictor import Predictor
from ViewsPredictor import ViewsPredictor
from STDPredictor import STDPredictor
from ItemBasedPredictor import ItemBasedPredictor
from SlopeOnePredictor import SlopeOnePredictor

from UserItemData import UserItemData
from MovieData import MovieData

import pandas as pd
import numpy as np
import math

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


    def evaluate(self, X:UserItemData, num):
        # Initial variables needed
        mae, mae_sum, num_of_calculations = [0]*3 # python magic hihi
        rmse, rmse_sum = 0, 0, 
        precision, precision_sum = 0, 0
        recall, recall_sum = 0, 0
        f = 0

        # start
        test_frame = X.user_ratings
        unique_users = test_frame["userID"].unique()
        unique_movies = test_frame["movieID"].unique()

        for user_id in unique_users:
            #print("USER:",user_id)
            predicted_frame = self.predictor.predict(user_id, num)
            if len(predicted_frame) == 0: 
                continue
            recommended_movies = predicted_frame[predicted_frame["userID"] == user_id]
            rated_movies = test_frame[test_frame["userID"]==user_id]
            #print(rated_movies)
            matching_movies = np.intersect1d( recommended_movies["movieID"].unique(), rated_movies["movieID"].unique() )
            if len(matching_movies) == 0: # No recommended movies were rated
                continue

            # Precision: len(intersection) / len(predicted) -> summed and averaged
            # Recall : len(intersection) / len(rated) -> summed and averaged
            # F: 2*P*R / (P+R)
            precision_sum += len(matching_movies) / len(recommended_movies)
            recall_sum += len(matching_movies) / len(rated_movies)

            for comparing_movie in matching_movies:
                rec_rating = list(recommended_movies.loc[recommended_movies["movieID"]==comparing_movie, "rec_rating"])[0]
                actual_rating = list(rated_movies.loc[rated_movies["movieID"]==comparing_movie, "rating"])[0]
                #print("Recommended:",rec_rating," Actual:",actual_rating)
                mae_sum += abs(actual_rating - rec_rating)
                rmse_sum += (actual_rating - rec_rating)**2
                num_of_calculations += 1

        # Evaluated everyone
        mae = mae_sum / num_of_calculations
        rmse = math.sqrt(rmse_sum / num_of_calculations)
        if type(self.predictor) == ViewsPredictor:
            mae = -1
            rmse = -1
        precision = precision_sum / num_of_calculations
        recall = recall_sum / num_of_calculations
        f = 2*precision*recall / (precision+recall)
        return (rmse, mae, precision, recall, f)


md = MovieData('data/movies.dat')
uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000, end_date='1.1.2008')
pred = SlopeOnePredictor()
rec = Recommender(pred)
rec.fit(uim)

rec_items = rec.recommend(78, n=15, rec_seen=True)
for idmovie, val in rec_items:
    try:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
    except Exception:
        print("Movie does not have any ratings")

#pred.print_most_similar_movies(md, 20)
# SIMILAR ITEMS TO THIS
#rec_items = rec.recommend_similar_items(4993, 10)
#print('Filmi podobni "The Lord of the Rings: The Fellowship of the Ring": ')
#for idmovie, val in rec_items:
#    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))

uim_test = UserItemData('data/user_ratedmovies.dat', min_ratings=200, start_date='2.1.2008')
rmse, mae, precision, recall, f = rec.evaluate(uim_test, 20)
print("Evaluation:")
print("rmse: {}, mae: {}, precision: {}, recall: {}, f: {}".format(rmse if rmse>=0 else "NA", mae if rmse>=0 else "NA",
                                                                    precision, recall, f))

