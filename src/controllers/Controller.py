from typing import Any, List, Tuple

from PIL import Image, ImageDraw

from src.application.models import IPredictor
from src.controllers.models import IControllerObserver


class Controller:
    _INPUT_BRUSH_SIZE = (0.5, 0.5)

    def __init__(self, predictor: IPredictor):
        self._predictor = predictor
        self._observers: List[IControllerObserver] = []

        self._input_dims = predictor.get_input_dims()
        self._input_image = Image.new(mode="RGB", size=self._input_dims, color="white")
        self._drawer = ImageDraw.Draw(self._input_image)

    def register(self, observer: IControllerObserver) -> None:
        self._observers.append(observer)

    def draw(self, event: Any, canvas_dims: Tuple[int, int]) -> None:
        canvas_coords: Tuple[int, int] = (event.x, event.y)

        scaling_factors = (
            self._input_dims[0] / canvas_dims[0],
            self._input_dims[1] / canvas_dims[1],
        )

        input_coords = (
            canvas_coords[0] * scaling_factors[0],
            canvas_coords[1] * scaling_factors[1],
        )

        canvas_brush_size = (
            int(self._INPUT_BRUSH_SIZE[0] / scaling_factors[0]),
            int(self._INPUT_BRUSH_SIZE[1] / scaling_factors[1]),
        )

        self._drawer.ellipse(
            (
                input_coords[0] - self._INPUT_BRUSH_SIZE[0],
                input_coords[1] - self._INPUT_BRUSH_SIZE[1],
                input_coords[0] + self._INPUT_BRUSH_SIZE[0],
                input_coords[1] + self._INPUT_BRUSH_SIZE[1],
            ),
            fill="black",
        )

        for obs in self._observers:
            obs.update_drawing(canvas_coords, canvas_brush_size)

    def clear(self) -> None:
        xy = (0, 0, self._input_dims[0] - 1, self._input_dims[1] - 1)
        self._drawer.rectangle(xy=xy, fill="white")

        for obs in self._observers:
            obs.update_clearing()

    def predict(self) -> None:
        self._predictor.predict_markup(self._input_image)
