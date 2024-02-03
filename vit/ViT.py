import torch
import torch.nn as nn
from torch import Tensor

from PatchEmbedding import PatchEmbedding
from Block import Block


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
        # shape: (n_samples, n_patches, embedding_dim)

        x = x + self.pos_embedding
        x = self.pos_drop(x)
        # shape: (n_samples, n_patches, embedding_dim)

        for block in self.blocks:
            x = block(x)
        # shape: (n_samples, n_patches, embedding_dim)

        x = self.norm(x)
        # shape: (n_samples, n_patches, embedding_dim)

        x = x.mean(1)
        # shape: (n_samples, 1, embedding_dim)

        x = x.squeeze()
        # shape: (n_samples, embedding_dim)

        x = self.head(x)
        # shape: (n_samples, n_classes)

        return x
