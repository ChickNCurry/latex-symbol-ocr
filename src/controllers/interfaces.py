from abc import ABC, abstractmethod
from typing import Tuple


class IControllerObserver(ABC):
    @abstractmethod
    def update_drawing(
        self, coords: Tuple[int, int], brush_size: Tuple[int, int]
    ) -> None:
        pass

    @abstractmethod
    def update_clearing(self) -> None:
        pass
