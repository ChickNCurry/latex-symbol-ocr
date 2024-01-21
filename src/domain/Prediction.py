from PIL.Image import Image


class Prediction:
    def __init__(
        self, class_label: int, markup: str, render: Image, probability: float
    ):
        self.class_label = class_label
        self.markup = markup
        self.render = render
        self.probability = probability
