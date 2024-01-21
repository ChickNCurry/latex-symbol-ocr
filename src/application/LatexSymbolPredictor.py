from typing import Tuple
from PIL.Image import Image

from src.domain.Prediction import Prediction
from src.domain.Predictions import Predictions
from src.application.interfaces import IClassifier
from src.application.interfaces import IMapper
from src.application.interfaces import IRenderer
from src.application.interfaces import IPredictor


class LatexSymbolPredictor(IPredictor):
    def __init__(
        self,
        classifier: IClassifier,
        mapper: IMapper,
        renderer: IRenderer,
        predictions: Predictions,
    ):
        self.classifier = classifier
        self.mapper = mapper
        self.renderer = renderer
        self.predictions = predictions

    def predict_markup(self, image: Image) -> None:
        class_labels, probabilities = self.classifier.classify(image, 3)
        markups = self.mapper.map_to_markup(class_labels)
        renders = self.renderer.render_markup(markups)
        assert len(class_labels) == len(markups) == len(renders)

        predictions = [
            Prediction(class_labels[i], markups[i], renders[i], probabilities[i])
            for i in range(len(class_labels))
        ]

        self.predictions.set_predictions(predictions)

    def get_input_dims(self) -> Tuple[int, int]:
        return self.classifier.get_input_dims()
