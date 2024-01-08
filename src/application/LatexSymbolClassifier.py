from typing import List, Tuple

import cv2
import torch
from sklearn.preprocessing import LabelEncoder
from torch import load, Tensor
from PIL.Image import Image
import numpy as np

from src.application.CNN import CNN
from src.application.IClassifier import IClassifier


class LatexSymbolClassifier(IClassifier):
    def __init__(self) -> None:
        self.model = CNN(1, 369)
        self.input_dims = self.model.get_input_dims()
        with open('src/data/model_state.pt', 'rb') as f:
            self.model.load_state_dict(load(f))
        self.model.eval()

        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load('src/data/classes.npy')

    def classify(self, image: Image, top_k: int) -> Tuple[List[int], List[float]]:
        tensor = self._convert_image_to_tensor(image)
        output = self.model(tensor)
        probability_tensor, class_label_tensor = torch.topk(output, top_k, 1)
        class_labels, probabilities = class_label_tensor[0].tolist(
        ), probability_tensor[0].tolist()
        decoded_class_labels = self.encoder.inverse_transform(class_labels)
        return list(map(int, decoded_class_labels)), probabilities

    def get_input_dims(self) -> tuple[int, int]:
        return self.input_dims

    def _convert_image_to_tensor(self, image: Image) -> Tensor:
        image_resized = image.resize(self.input_dims)
        image_rgb = image_resized.convert("RGB")
        image_np = np.array(image_rgb)
        image_cv = image_np[:, :, ::-1].copy()
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        image_tensor = torch.FloatTensor(image_gray)[None, None, :, :]
        return image_tensor
