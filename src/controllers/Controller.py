from typing import List
from PIL import Image, ImageDraw

from src.controllers.IControllerObserver import IControllerObserver
from src.application.IPredictor import IPredictor


class Controller:
    def __init__(self, predictor: IPredictor):
        self.predictor = predictor
        self.input_dims = predictor.get_input_dims()
        self.observers: List[IControllerObserver] = []

        self.INPUT_BRUSH_SIZE = (0.5, 0.5)

        self.input_image = Image.new(mode="RGB", size=self.input_dims, color="white")
        self.drawer = ImageDraw.Draw(self.input_image)

    def register(self, observer: IControllerObserver) -> None:
        self.observers.append(observer)

    def draw(self, event, canvas_dims: tuple[int, int]) -> None:
        canvas_coords: tuple[int, int] = (event.x, event.y)
        scaling_factors = tuple(i / c for i, c in zip(self.input_dims, canvas_dims))
        input_coords = tuple(c * s for c, s in zip(canvas_coords, scaling_factors))
        canvas_brush_size = tuple(int(i / s) for i, s in zip(self.INPUT_BRUSH_SIZE, scaling_factors))

        self.drawer.ellipse((input_coords[0] - self.INPUT_BRUSH_SIZE[0],
                             input_coords[1] - self.INPUT_BRUSH_SIZE[1],
                             input_coords[0] + self.INPUT_BRUSH_SIZE[0],
                             input_coords[1] + self.INPUT_BRUSH_SIZE[1]), fill="black")

        for obs in self.observers:
            obs.update_drawing(canvas_coords, tuple[int, int](canvas_brush_size))

    def clear(self) -> None:
        self.drawer.rectangle((0, 0, self.input_dims[0] - 1, self.input_dims[1] - 1), fill="white")
        for obs in self.observers:
            obs.update_clearing()

    def predict(self) -> None:
        self.predictor.predict_markup(self.input_image)
