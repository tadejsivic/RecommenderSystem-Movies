import datetime
import csv
from MovieData import MovieData

import numpy as np
import pandas as pd

class UserItemData:
    def __init__(self, path, start_date = datetime.datetime.min, end_date = datetime.datetime.max, min_ratings = 0):
        self.path = path
        self.start_date = datetime.datetime.strptime(start_date, "%d.%m.%Y") if isinstance(start_date, str) else start_date
        self.end_date = datetime.datetime.strptime(end_date, "%d.%m.%Y") if isinstance(end_date, str) else end_date
        self.min_ratings = min_ratings
        self.user_ratings = []
        self.movies = MovieData("data/movies.dat")
        self.load_data()

    def load_data(self):
        with open(self.path) as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader)
            for row in reader:
                rating_date = datetime.datetime(year=int(row[5]), month=int(row[4]), day=int(row[3]))
                if self.start_date <= rating_date <= self.end_date and self.movies.get_nratings(int(row[1])) >= self.min_ratings:
                    self.user_ratings.append(row)

    def nratings(self):
        return len(self.user_ratings)

ui_data = UserItemData("data/user_ratedmovies.dat", start_date = "12.1.2007", end_date="16.2.2008", min_ratings=100)
print(ui_data.nratings())