from typing import Tuple
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer  # type: ignore
from torch import Tensor
from transformers import PreTrainedTokenizerFast  # type: ignore


class Encoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(Encoder, self).__init__()

        self.pos_enc_2d_summer = Summer(PositionalEncodingPermute2D(hidden_size))

        self.sequential = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(256, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.sequential(x)
        x = self.pos_enc_2d_summer(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        super(Decoder, self).__init__()

        self.tokenizer = tokenizer
        self.embed = nn.Embedding(tokenizer.vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward_step(
        self, input_token_id: Tensor, states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        embedding = self.embed(input_token_id)
        embedding = self.dropout(embedding)
        print(embedding.shape)

        output, states = self.lstm(embedding, states)
        logits = self.linear(output)
        print(logits.shape)

        id = logits.argmax(1)
        return id, states

    def forward(
        self,
        features: Tensor,
        target_token_ids: Tensor | None = None,
        max_len: int = 50,
    ) -> str:
        batch_size = features.shape[0]
        input_token_id = torch.empty(batch_size, 1).fill_(
            self.tokenizer.token_to_id("[SOS]")
        )
        states = (features.mean(2), features.mean(2))
        output_token_ids = []

        for i in range(max_len):
            output_token_id, states = self.forward_step(input_token_id, states)
            output_token_ids.append(output_token_id)

            if target_token_ids is not None:
                input_token_id = target_token_ids[:, i].unsqueeze(1)
            else:
                input_token_id = output_token_id.detach()

            if self.tokenizer.id_to_token(output_token_id) == "[EOS]":
                break

        equation = str(self.tokenizer.decode(output_token_ids))
        return equation


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        super(EncoderDecoder, self).__init__()

        self.tokenizer = tokenizer
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(embed_size, hidden_size, num_layers, tokenizer)

    def forward(
        self,
        img: Tensor,
        target_token_ids: Tensor | None = None,
        max_len: int = 50,
    ) -> str:
        features = self.encoder(img)
        equation = self.decoder(features, target_token_ids, max_len)
        return str(equation)
