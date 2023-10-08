from typing import List
from PIL import Image, ImageDraw

from src.controllers.IControllerObserver import IControllerObserver
from src.application.IPredictor import IPredictor


class Controller:

    def __init__(self, predictor: IPredictor):
        self.predictor = predictor
        self.observers: List[IControllerObserver] = []

        self.input_dims = predictor.get_input_dims()
        self.input_brush_size = 0.5

        self.input_image = Image.new(mode="RGB", size=self.input_dims, color="white")
        self.drawer = ImageDraw.Draw(self.input_image)

    def register(self, observer: IControllerObserver):
        self.observers.append(observer)

    def draw(self, event, canvas_dims: tuple[int, int]):
        scaling_factors = tuple(input_dim / canvas_dim for input_dim, canvas_dim in zip(self.input_dims, canvas_dims))

        canvas_x, canvas_y = event.x, event.y
        input_x, input_y = canvas_x * scaling_factors[0], canvas_y * scaling_factors[1]
        canvas_brush_size_x = int(self.input_brush_size / scaling_factors[0])
        canvas_brush_size_y = int(self.input_brush_size / scaling_factors[1])

        self.drawer.ellipse((input_x - self.input_brush_size,
                             input_y - self.input_brush_size,
                             input_x + self.input_brush_size,
                             input_y + self.input_brush_size), fill="black")

        for obs in self.observers:
            obs.update_drawing(canvas_x, canvas_y, (canvas_brush_size_x, canvas_brush_size_y))

    def clear(self):
        self.drawer.rectangle((0, 0, self.input_dims[0] - 1, self.input_dims[1] - 1), fill="white")
        for obs in self.observers:
            obs.update_clearing()

    def predict(self):
        # self.image.show()
        self.predictor.predict(self.input_image)
