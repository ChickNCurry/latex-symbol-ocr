from abc import ABC, abstractmethod
from typing import List, Tuple

from PIL.Image import Image
from PIL.Image import Image


class IInputDependant(ABC):
    @abstractmethod
    def get_input_dims(self) -> Tuple[int, int]:
        pass


class IPredictor(IInputDependant):
    @abstractmethod
    def predict_markup(self, image: Image) -> None:
        pass


class IClassifier(IInputDependant):
    @abstractmethod
    def classify(self, image: Image, top_k: int) -> Tuple[List[int], List[float]]:
        pass


class IMapper(ABC):
    @abstractmethod
    def map_to_markup(self, class_labels: List[int]) -> List[str]:
        pass


class IRenderer(ABC):
    @abstractmethod
    def render_markup(self, markups: List[str]) -> List[Image]:
        pass
