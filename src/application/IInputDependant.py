from abc import ABC, abstractmethod


class IInputDependant(ABC):

    @abstractmethod
    def get_input_dims(self) -> tuple[int, int]:
        pass
