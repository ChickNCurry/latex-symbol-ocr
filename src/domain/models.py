from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL.Image import Image
from typing import List


@dataclass
class Prediction:
    class_label: int
    markup: str
    render: Image
    probability: float


class IPredictionsObserver(ABC):
    @abstractmethod
    def update_predictions(self, predictions: List[Prediction]) -> None:
        pass
