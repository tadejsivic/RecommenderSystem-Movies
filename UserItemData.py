import datetime
from MovieData import MovieData

import numpy as np
import pandas as pd

class UserItemData:
    def __init__(self, path, start_date = pd.Timestamp.min, end_date = pd.Timestamp.max, min_ratings = 0):
        self.path = path
        self.start_date = pd.to_datetime(start_date, dayfirst=True)
        self.end_date = pd.to_datetime(end_date, dayfirst=True)
        self.min_ratings = min_ratings
        #self.movies = MovieData("data/movies.dat")
        self.load_data()


    def load_data(self):
        self.user_ratings = pd.read_csv(self.path, sep="\t", parse_dates={"date" : ["date_year", "date_month", "date_day"]})
        mask = (self.start_date <= self.user_ratings["date"]) & (self.user_ratings["date"] < self.end_date)
        self.user_ratings = self.user_ratings[mask]
        self.user_ratings = self.user_ratings.groupby("movieID").filter(lambda x : len(x) >= self.min_ratings)


    def nratings(self):
        return len(self.user_ratings)


#ui_data = UserItemData("data/user_ratedmovies.dat", start_date = "12.1.2007", end_date="16.2.2008", min_ratings=100)
#print(ui_data.nratings())
#print(ui_data.movies.get_title(1))