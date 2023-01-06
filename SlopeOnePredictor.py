from Predictor import Predictor

from UserItemData import UserItemData
from MovieData import MovieData

import numpy as np
import pandas as pd

# Prediktor, ki napove filme, ki jih uporabnik še ni gledal
# Za merjenje podobnosti se uporabijo trojčki, oz. metoda Slope One
# To pomeni, koliko bi uporabnik ocenil film, glede na njegovo povprečje in kako se filmi ocenjejujo povprečno med seboj

# Zagnati z rec_seen=True

class SlopeOnePredictor(Predictor):

    # Compute all deviations for all pairs of movies.
    def fit(self, X:UserItemData):
        self.frame = pd.DataFrame()
        self.frame["userID"] = X.user_ratings["userID"]
        self.frame["movieID"] = X.user_ratings["movieID"]
        self.frame["rating"] = X.user_ratings["rating"]

        self.deviations = {}
        self.num_of_calculations = {}
        unique_movies = self.frame["movieID"].unique()
        for p1 in unique_movies:
            for p2 in unique_movies:
                if p1 == p2:
                    continue
                if (p1,p2) in self.deviations: 
                    self.deviations[(p2,p1)] = -self.deviations[(p1,p2)]
                    self.num_of_calculations[(p2,p1)] = self.num_of_calculations[(p1,p2)]
                    continue
                elif (p2,p1) in self.deviations:
                    self.deviations[(p1,p2)] = -self.deviations[(p2,p1)]
                    self.num_of_calculations[(p1,p2)] = self.num_of_calculations[(p2,p1)]
                    continue

                # Compute deviation for p1,p2
                first_movie = self.frame[self.frame["movieID"] == p1]
                second_movie = self.frame[self.frame["movieID"] == p2]
                users_both_movies = np.intersect1d( first_movie["userID"].unique(), second_movie["userID"].unique() )
                if len(users_both_movies) == 0:
                    self.deviations[(p1,p2)] = -10
                    self.num_of_calculations[(p1,p2)] = 0
                    continue
                first_movie = first_movie.loc[first_movie["userID"].isin(users_both_movies)]
                second_movie = second_movie.loc[second_movie["userID"].isin(users_both_movies)]
                first_movie_ratings = np.array(first_movie["rating"].tolist())
                second_movie_ratings = np.array(second_movie["rating"].tolist())
                
                diff = first_movie_ratings - second_movie_ratings # or the other way around ???
                dev = np.sum(diff) / len(diff)
                self.deviations[(p1,p2)] = dev
                self.num_of_calculations[(p1,p2)] = len(diff)


    # Use the deviations computed in fit, to predict movies for a target user
    def predict(self, user_id, n):
        users_average = np.array(self.frame.loc[self.frame["userID"] == user_id, "rating"])
        if len(users_average) == 0: return pd.DataFrame()
        users_average = users_average.mean()
        user_rated_movies = self.frame.loc[self.frame["userID"] == user_id, "movieID"].tolist()
        unique_movies = self.frame["movieID"].unique()
        movies_to_recommend = [movie_id for movie_id in unique_movies if movie_id not in user_rated_movies]

        temp_frame = pd.DataFrame()
        temp_frame["movieID"] = movies_to_recommend
        temp_frame["userID"] = user_id
        temp_frame["rec_rating"] = -1

        for target_movie in movies_to_recommend:
            total_sum = 0
            total_calculations = 0
            for rated_movie in user_rated_movies:
                rating = self.frame.loc[(self.frame["userID"]==user_id) & (self.frame["movieID"]==rated_movie), "rating"].tolist()[0]
                deviation = self.get_deviation(target_movie, rated_movie)
                if deviation == -10:
                    continue
                total_sum += (rating + deviation)*self.get_num_of_calculations(target_movie, rated_movie)
                total_calculations += self.get_num_of_calculations(target_movie, rated_movie)
            if total_calculations != 0:
                rec = total_sum / total_calculations
            else:
                rec = users_average
            temp_frame.loc[temp_frame["movieID"]==target_movie, "rec_rating"] = rec
        return temp_frame


    def get_deviation(self, p1, p2):
        if p1==p2: return -10
        return self.deviations[(p1,p2)]

    def get_num_of_calculations(self, p1, p2):
        if p1==p2: return 0
        return self.num_of_calculations[(p1,p2)]


#sop = SlopeOnePredictor()
#sop.fit(UserItemData("data/user_ratedmovies.dat", min_ratings=1000))
#print(sop.predict(190, 50))
