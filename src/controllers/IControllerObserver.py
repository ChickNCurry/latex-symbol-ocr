from abc import ABC, abstractmethod


class IControllerObserver(ABC):

    @abstractmethod
    def update_drawing(self, coords: tuple[int, int], brush_size: tuple[int, int]):
        pass

    @abstractmethod
    def update_clearing(self):
        pass
