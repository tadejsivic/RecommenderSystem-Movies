from abc import abstractmethod, ABC

# Abstrakten razred, ki je nadrazred vsem prediktorjem. Uporabno le zaradi metod in recommenderja

class Predictor(ABC):
    def fit(self, X):
        pass
    def predict(self, user_id, n):
        pass