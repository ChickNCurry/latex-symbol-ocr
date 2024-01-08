from abc import ABC, abstractmethod
from typing import Tuple


class IInputDependant(ABC):

    @abstractmethod
    def get_input_dims(self) -> Tuple[int, int]:
        pass
