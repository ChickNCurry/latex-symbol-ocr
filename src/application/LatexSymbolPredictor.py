from PIL import Image

from src.application.IMapper import IMapper
from src.domain.Prediction import Prediction
from src.domain.Predictions import Predictions
from src.application.IClassifier import IClassifier
from src.application.IRenderer import IRenderer
from src.application.IPredictor import IPredictor


class LatexSymbolPredictor(IPredictor):
    def __init__(self, classifier: IClassifier, mapper: IMapper, renderer: IRenderer, predictions: Predictions):
        self.classifier = classifier
        self.mapper = mapper
        self.renderer = renderer
        self.predictions = predictions

    def predict(self, image: Image) -> None:
        (class_labels, probabilities) = self.classifier.classify(image, 3)
        markups = self.mapper.map(class_labels)
        renders = self.renderer.render(markups)
        assert len(class_labels) == len(markups) == len(renders)

        predictions = []
        for i in range(len(class_labels)):
            prediction = Prediction(class_labels[i], markups[i], renders[i], probabilities[i])
            predictions.append(prediction)
        self.predictions.set_predictions(predictions)

    def get_input_dims(self) -> tuple:
        return self.classifier.get_input_dims()
