import string
from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image

from src.domain.Prediction import Prediction


class IPredictionsObserver(ABC):

    @abstractmethod
    def update_predictions(self, predictions: List[Prediction]):
        pass
