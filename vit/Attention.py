import torch.nn as nn
from torch import Tensor


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        qkv_bias: bool = True,
        attn_p: float = 0.0,
        proj_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale: float = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x: Tensor) -> Tensor:
        # x and returns shape: (n_samples, n_patches, dim)

        n_samples, n_patches, dim = x.shape
        assert dim == self.dim

        qkv: Tensor = self.qkv(x)
        # shape: (n_samples, n_patches, dim * 3)

        qkv = qkv.reshape(n_samples, n_patches, 3, self.n_heads, self.head_dim)
        # shape: (n_samples, n_patches, 3, n_heads, head_dim)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        # shape: (3, n_samples, n_heads, n_patches, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        # each shape: (n_samples, n_heads, n_patches, head_dim)

        k_t = k.transpose(-2, -1)
        # shape: (n_samples, n_heads, head_dim, n_patches)

        dp = (q @ k_t) * self.scale
        # shape: (n_samples, n_heads, n_patches, n_patches)

        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # shape: (n_samples, n_heads, n_patches, n_patches)

        weighted_avg = attn @ v
        # shape: (n_samples, n_heads, n_patches, head_dim)

        weighted_avg = weighted_avg.transpose(1, 2)
        # shape: (n_samples, n_patches, n_heads, head_dim)

        weighted_avg = weighted_avg.flatten(2)
        # shape: (n_samples, n_patches, dim)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        # shape: (n_samples, n_patches, dim)

        return x
