from abc import abstractmethod, ABC

class Predictor(ABC):
    def fit(self, X):
        pass
    def predict(self, user_id, n):
        pass