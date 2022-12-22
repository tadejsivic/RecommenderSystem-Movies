import csv
import numpy as np
import pandas as pd

class MovieData:
    def __init__(self, path):
        self.path = path
        self.movies = {}
        self.load_data()

    def load_data(self):
        with open(self.path, encoding="latin1") as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader)
            for row in reader:
                self.movies[int(row[0])] = row

    def get_title(self, movie_id):
        return self.movies[movie_id][1]

