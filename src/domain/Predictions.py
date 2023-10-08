from typing import List

from src.domain.IPredictionsObserver import IPredictionsObserver
from src.domain.Prediction import Prediction


class Predictions:
    def __init__(self):
        self.predictions: List[Prediction] = []
        self.observers: List[IPredictionsObserver] = []

    def register(self, observer: IPredictionsObserver):
        self.observers.append(observer)

    def set_predictions(self, predictions: List[Prediction]):
        self.predictions = predictions
        for obs in self.observers:
            obs.update_predictions(self.predictions)
