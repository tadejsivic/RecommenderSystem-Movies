import csv

class MovieData:
    def __init__(self, path):
        self.path = path
        self.movies = {}
        self.load_data()

    def load_data(self):
        with open(self.path) as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader)
            for row in reader:
                self.movies[int(row[0])] = row

    def get_title(self, movie_id):
        return self.movies[movie_id][1]
    
    def get_nratings(self, movie_id):
        nratings = self.movies[movie_id][18]
        if nratings.isnumeric():
            return int(nratings)
        return 0


