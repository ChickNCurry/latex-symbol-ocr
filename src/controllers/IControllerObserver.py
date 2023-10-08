from abc import ABC, abstractmethod

from PIL.Image import Image


class IControllerObserver(ABC):

    @abstractmethod
    def update_drawing(self, x, y, brush_size: tuple[int, int]):
        pass

    @abstractmethod
    def update_clearing(self):
        pass
