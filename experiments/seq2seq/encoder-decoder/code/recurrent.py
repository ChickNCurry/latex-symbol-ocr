from typing import Tuple, Union
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import (  # type: ignore
    PositionalEncodingPermute2D,
    Summer,
)
from torch import Tensor
from transformers import PreTrainedTokenizerFast  # type: ignore


class Encoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(Encoder, self).__init__()

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
        feature_maps: Tensor = self.sequential(x)
        return feature_maps


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device,
    ) -> None:
        super(Decoder, self).__init__()

        self.device = device
        self.tokenizer = tokenizer
        self.embed = nn.Embedding(tokenizer.vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, device=device, num_layers=12)
        self.linear = nn.Linear(hidden_size, tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward_step(
        self, input_token_id: Tensor, states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        embedding = self.dropout(self.embed(input_token_id))

        # # print(states[0].shape)
        # # print(embedding.shape)

        output, states = self.lstm(embedding, states)
        logits = self.linear(output)

        # # print(logits.shape)

        return logits, states

    def forward(
        self,
        states: Tuple[Tensor, Tensor],
        target_token_ids: Union[Tensor, None] = None,
        max_len: int = 200,
    ) -> Tensor:

        max_len = min(
            max_len, target_token_ids.shape[1] if target_token_ids is not None else max_len
        )

        # print(
        #    "target_token_ids",
        #    target_token_ids.shape if target_token_ids is not None else None,
        # )

        batch_size = states[0].shape[0]

        input_token_id = (
            torch.empty(batch_size, 1, dtype=torch.int64)
            .fill_(self.tokenizer.convert_tokens_to_ids("[BOS]"))
            .to(self.device)
        )

        # print("input_token_id", input_token_id.shape)

        # shape: (num_layers, batch_size, hidden_size)
        states = (states[0].repeat(12, 1, 1), states[1].repeat(12, 1, 1))

        logits_list = []

        for i in range(max_len):
            # print("states", states[0].shape, states[1].shape)

            logits, states = self.forward_step(input_token_id, states)

            # print("states", states[0].shape, states[1].shape)
            # print("logits", logits.shape)

            logits_list.append(logits)

            output_token_id = logits.argmax(2)
            # print("output_token_id", output_token_id.shape)

            if target_token_ids is not None:
                # teacher forcing
                input_token_id = target_token_ids[:, i].repeat(12, 1)
            else:
                input_token_id = output_token_id

            # print("input_token_id", input_token_id.shape)

            ids = input_token_id.squeeze(1).tolist()
            # print("ids", ids)

            tokens = self.tokenizer.convert_ids_to_tokens(ids)

            # print("tokens", tokens)

            if all(token == "[EOS]" for token in tokens):
                break

        output = torch.cat(logits_list, dim=1)

        # print("output", output.shape, output.dtype)

        return output


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device,
    ) -> None:
        super(EncoderDecoder, self).__init__()

        self.device = device
        self.tokenizer = tokenizer
        self.encoder = Encoder(hidden_size)
        self.pos_enc_2d_summer = Summer(PositionalEncodingPermute2D(hidden_size))
        self.decoder = Decoder(embed_size, hidden_size, tokenizer, device)

    def forward(
        self,
        img: Tensor,
        target_token_ids: Union[Tensor, None] = None,
        max_len: int = 50,
    ) -> Tensor:
        feature_maps = self.encoder(img)
        feature_maps = self.pos_enc_2d_summer(feature_maps)

        # print("feature_maps", feature_maps.shape)

        features = feature_maps.view(feature_maps.shape[0], feature_maps.shape[1], -1)

        # print("features", features.shape)

        states = (features.mean(2), features.mean(2))

        # print("states", states[0].shape, states[1].shape)

        logits_tensor: Tensor = self.decoder(states, target_token_ids, max_len)
        return logits_tensor
