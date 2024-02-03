import torch.nn as nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        emb_dim: int = 768,
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
