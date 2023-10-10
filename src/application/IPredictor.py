from abc import abstractmethod

from PIL.Image import Image

from src.application.IInputDependant import IInputDependant


class IPredictor(IInputDependant):

    @abstractmethod
    def predict_markup(self, image: Image):
        pass
