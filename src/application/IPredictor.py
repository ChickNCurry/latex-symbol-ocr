from abc import ABC, abstractmethod

from PIL.Image import Image


class IPredictor(ABC):

    @abstractmethod
    def predict_markup(self, image: Image):
        pass

    @abstractmethod
    def get_input_dims(self) -> tuple[int, int]:
        pass
