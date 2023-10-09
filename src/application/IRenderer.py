from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image


class IRenderer(ABC):

    @abstractmethod
    def render_markup(self, markups: List[str]) -> List[Image]:
        pass
