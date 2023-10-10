import torch.nn as nn
from torch import Tensor

from src.application.IInputDependant import IInputDependant


class CNN(nn.Module, IInputDependant):
    def __init__(self, in_channels, num_of_classes):
        super().__init__()

        self.input_dims = (32, 32)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 5 * 5, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_of_classes)
        )

        # 1 * 32 * 32
        # 32 * 28 * 28
        # 32 * 14 * 14
        # 64 * 10 * 10
        # 64 * 5 * 5

        self._init_weights()

    def forward(self, x) -> Tensor:
        return self.model(x)

    def get_input_dims(self) -> tuple[int, int]:
        return self.input_dims

    def _init_weights(self) -> None:
        for m in self.model:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)