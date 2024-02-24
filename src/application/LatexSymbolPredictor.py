from typing import Tuple

from PIL.Image import Image

from src.application.models import IClassifier, IMapper, IPredictor, IRenderer
from src.domain.models import Prediction
from src.domain.Predictions import Predictions


class LatexSymbolPredictor(IPredictor):
    def __init__(
        self,
        classifier: IClassifier,
        mapper: IMapper,
        renderer: IRenderer,
        predictions: Predictions,
    ):
        self._classifier = classifier
        self._mapper = mapper
        self._renderer = renderer
        self._predictions = predictions

    def predict_markup(self, image: Image) -> None:
        class_labels, probabilities = self._classifier.classify(image, 3)

        markups = self._mapper.map_to_markup(class_labels)
        renders = self._renderer.render_markup(markups)

        assert len(class_labels) == len(markups) == len(renders)

        predictions = [
            Prediction(*z) for z in zip(class_labels, markups, renders, probabilities)
        ]

        self._predictions.set_predictions(predictions)

    def get_input_dims(self) -> Tuple[int, int]:
        return self._classifier.get_input_dims()
