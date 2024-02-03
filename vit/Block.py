import torch.nn as nn
from torch import Tensor

from Attention import Attention
from MLP import MLP


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        p: float = 0.0,
        attn_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        # x and returns shape: (n_samples, n_patches, dim)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
