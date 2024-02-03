import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, features: int, hidden_features: int, p: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, features)
        self.drop = nn.Dropout(p)

    def forward(self, x: Tensor) -> Tensor:
        # x and returns shape: (n_samples, features)

        features = x.shape[-1]
        assert features == self.fc1.in_features

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # shape: (n_samples, hidden_features)

        x = self.fc2(x)
        x = self.drop(x)
        # shape: (n_samples, features)

        return x
