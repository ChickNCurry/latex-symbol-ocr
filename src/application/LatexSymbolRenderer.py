import io
from typing import List

from PIL import Image
import matplotlib.pyplot as plt

from src.application.IRenderer import IRenderer


class LatexSymbolRenderer(IRenderer):
    def render(self, markups: List[str]) -> List[Image]:
        images = []

        for markup in markups:
            plt.figure()
            plt.axis("off")
            plt.text(0.5, 0.5, f"${markup}$", size=200, ha="center", va="center")

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')

            image = Image.open(img_buf)
            images.append(image)

        return images
