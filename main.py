from src.application.LatexSymbolClassifier import LatexSymbolClassifier
from src.application.LatexSymbolMapper import LatexSymbolMapper
from src.application.LatexSymbolPredictor import LatexSymbolPredictor
from src.application.LatexSymbolRenderer import LatexSymbolRenderer
from src.controllers.Controller import Controller
from src.domain.Predictions import Predictions
from src.presentation.View import View

predictions = Predictions()

classifier = LatexSymbolClassifier()
mapper = LatexSymbolMapper()
renderer = LatexSymbolRenderer()
predictor = LatexSymbolPredictor(classifier, mapper, renderer, predictions)

controller = Controller(predictor)
view = View(controller)

controller.register(view)
predictions.register(view)

view.run()
