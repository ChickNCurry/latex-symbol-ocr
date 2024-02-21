import torch
import torch.nn as nn
from torch import Tensor

# https://www.youtube.com/watch?v=ovB0ddFtzzA&t=71s


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (n_samples, in_channels, img_size, img_size)
        # returns shape: (n_samples, n_patches, emb_dim)

        _, c, h, w = x.shape
        assert c == self.proj.in_channels
        assert h == self.img_size
        assert w == self.img_size

        x = self.proj(x)
        # shape: (n_samples, emb_dim, n_patches ** 0.5, n_patches ** 0.5)
        # note: n_patches ** 0.5 = img_size // patch_size

        x = x.flatten(2)
        # shape: (n_samples, emb_dim, n_patches)

        x = x.transpose(1, 2)
        # shape: (n_samples, n_patches, emb_dim)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        qkv_bias: bool,
        attn_p: float,
        proj_p: float,
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


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, p: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(p)

    def forward(self, x: Tensor) -> Tensor:
        # x and returns shape: (n_samples, n_patches, dim)

        features = x.shape[-1]
        assert features == self.fc1.in_features

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # shape: (n_samples, n_patches, hidden_dim)

        x = self.fc2(x)
        x = self.drop(x)
        # shape: (n_samples, n_patches, dim)

        return x


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
        self.mlp = MLP(dim, int(dim * mlp_ratio), p=p)

    def forward(self, x: Tensor) -> Tensor:
        # x and returns shape: (n_samples, n_patches, dim)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        n_classes: int = 369,
        emb_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        p: float = 0.0,
        attn_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, in_channels, emb_dim
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embedding.n_patches, emb_dim)
        )
        self.pos_drop = nn.Dropout(p)
        self.blocks = nn.ModuleList(
            [
                Block(emb_dim, n_heads, mlp_ratio, qkv_bias, p, attn_p)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (n_samples, in_channels, img_size, img_size)
        # returns shape: (n_samples, n_classes)

        x = self.patch_embedding(x)
        # shape: (n_samples, n_patches, emb_dim)

        x = x + self.pos_embedding
        x = self.pos_drop(x)
        # shape: (n_samples, n_patches, emb_dim)

        for block in self.blocks:
            x = block(x)
        # shape: (n_samples, n_patches, emb_dim)

        x = self.norm(x)
        # shape: (n_samples, n_patches, emb_dim)

        x = x.mean(1)
        # shape: (n_samples, 1, emb_dim)

        x = x.squeeze()
        # shape: (n_samples, emb_dim)

        x = self.head(x)
        # shape: (n_samples, n_classes)

        return x
