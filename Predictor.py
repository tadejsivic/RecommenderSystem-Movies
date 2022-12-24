from abc import abstractmethod, ABC

class Predictor(ABC):
    def fit(self):
        pass
    def predict(self):
        pass