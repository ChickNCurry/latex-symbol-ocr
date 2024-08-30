import math
import torch
from torch import Tensor, nn
from torchvision.models import swin_t, vgg11, Swin_T_Weights, VGG11_Weights  # type: ignore
from positional_encodings.torch_encodings import (  # type: ignore
    Summer,
    PositionalEncoding1D,
    PositionalEncodingPermute2D,
)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float) -> None:
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_hidden)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # (batch_size, seq_len, d_model)
        x = self.fc2(x)

        return x


# class CNN(nn.Module):
#     def __init__(self, d_feature: int) -> None:
#         super(CNN, self).__init__()

#         self.sequential = nn.Sequential(
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(1, 2),
#             nn.Conv2d(256, d_feature, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(d_feature),
#             nn.MaxPool2d(2, 1),
#             nn.Conv2d(d_feature, d_feature, 3, padding=1),
#             nn.BatchNorm2d(d_feature),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.sequential(x)
#         return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super(EncoderBlock, self).__init__()

        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.mlp = MLP(d_model, d_hidden, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model)
        output, _ = self.mha(x, x, x, need_weights=False)
        x = x + self.dropout1(output)
        x = self.norm1(x)

        # (batch_size, seq_len, d_model)
        output = self.mlp(x)
        x = x + self.dropout2(output)
        x = self.norm2(x)

        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        d_feature: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super(ImageEncoder, self).__init__()

        # self.cnn = CNN(d_feature)
        self.cnn = nn.Sequential(*list(vgg11(weights=VGG11_Weights.DEFAULT).features))

        self.pos_enc_2d_summer = Summer(PositionalEncodingPermute2D(d_feature))
        self.proj = nn.Linear(d_feature, d_model)

        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_hidden, dropout) for _ in range(n_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, 1, height, width)

        # (batch_size, 3, height, width)
        x = x.repeat(1, 3, 1, 1)

        # (batch_size, d_feature, height, width)
        x = self.cnn(x)  # * math.sqrt(self.d_model)
        x = self.pos_enc_2d_summer(x)

        # (batch_size, d_feature, height * width = seq_len)
        x = x.view(x.shape[0], x.shape[1], -1)

        # (batch_size, seq_len, d_feature)
        x = x.permute(0, 2, 1)

        # (batch_size, seq_len, d_model)
        x = self.proj(x)

        # (batch_size, seq_len, d_model)
        for block in self.blocks:
            x = block(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super(DecoderBlock, self).__init__()

        self.mha1 = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.mha2 = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = MLP(d_model, d_hidden, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        causal_mask: Tensor,
        encoder_output: Tensor,
    ) -> Tensor:
        # x shape: (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model)
        output, _ = self.mha1(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=False,
            attn_mask=causal_mask,
            is_causal=True,
        )
        x = x + self.dropout1(output)
        x = self.norm1(x)

        # (batch_size, seq_len, d_model)
        output, _ = self.mha2(
            x,
            encoder_output,
            encoder_output,
            need_weights=False,
        )

        x = x + self.dropout2(output)
        x = self.norm2(x)

        # (batch_size, seq_len, d_model)
        output = self.mlp(x)
        x = x + self.dropout3(output)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super(Decoder, self).__init__()

        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc_1d_summer = Summer(PositionalEncoding1D(d_model))

        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_hidden, dropout) for _ in range(n_blocks)]
        )

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        causal_mask: Tensor,
        encoder_output: Tensor,
    ) -> Tensor:
        # x shape: (batch_size, seq_len)

        # (batch_size, seq_len, d_model)
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_enc_1d_summer(x)

        # (batch_size, seq_len, d_model)
        for block in self.blocks:
            x = block(x, padding_mask, causal_mask, encoder_output)

        return x


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_feature: int,
        d_model: int,
        n_blocks_enc: int,
        n_blocks_dec: int,
        n_heads: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super(EncoderDecoder, self).__init__()

        # self.encoder = ImageEncoder(
        #     d_feature, d_model, n_blocks_enc, n_heads, d_hidden, dropout
        # )
        self.encoder = swin_t(weights=Swin_T_Weights.DEFAULT).features
        self.proj = nn.Linear(d_feature, d_model)  # only with swin

        self.decoder = Decoder(
            vocab_size, d_model, n_blocks_dec, n_heads, d_hidden, dropout
        )
        self.fc = nn.Linear(d_model, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: Tensor, y: Tensor, padding_mask: Tensor, causal_mask: Tensor
    ) -> Tensor:
        # x shape: (batch_size, seq_len)
        # y shape: (batch_size, seq_len)

        # (batch_size, 3, height, width)
        x = x.repeat(1, 3, 1, 1)  # only for swin

        # (batch_size, seq_len, d_model)
        encoder_output = self.encoder(x)
        encoder_output = encoder_output.view(
            encoder_output.shape[0],
            encoder_output.shape[1] * encoder_output.shape[2],
            encoder_output.shape[3],
        )
        encoder_output = self.proj(encoder_output)  # only for swin

        # (batch_size, seq_len, d_model)
        decoder_output = self.decoder(y, padding_mask, causal_mask, encoder_output)

        # (batch_size, seq_len, vocab_size)
        logits: Tensor = self.fc(decoder_output)

        # (batch_size, vocab_size, seq_len)
        logits = logits.permute(0, 2, 1)

        return logits
