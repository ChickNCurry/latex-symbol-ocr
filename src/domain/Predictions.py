from typing import List

from src.domain.models import IPredictionsObserver, Prediction


class Predictions:
    def __init__(self) -> None:
        self._predictions: List[Prediction] = []
        self._observers: List[IPredictionsObserver] = []

    def register(self, observer: IPredictionsObserver) -> None:
        self._observers.append(observer)

    def set_predictions(self, predictions: List[Prediction]) -> None:
        self._predictions = predictions
        for obs in self._observers:
            obs.update_predictions(self._predictions)
