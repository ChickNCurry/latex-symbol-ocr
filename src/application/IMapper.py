from abc import ABC, abstractmethod
from typing import List


class IMapper(ABC):

    @abstractmethod
    def map(self, class_labels: List[int]) -> List[str]:
        pass
