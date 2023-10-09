from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image


class IClassifier(ABC):

    @abstractmethod
    def classify(self, image: Image, top_k: int) -> (List[int], List[float]):
        pass

    @abstractmethod
    def get_input_dims(self) -> tuple[int, int]:
        pass
