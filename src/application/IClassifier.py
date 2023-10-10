from abc import abstractmethod
from typing import List

from PIL.Image import Image

from src.application.IInputDependant import IInputDependant


class IClassifier(IInputDependant):

    @abstractmethod
    def classify(self, image: Image, top_k: int) -> tuple[List[int], List[float]]:
        pass
