from typing import Tuple, Union
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer  # type: ignore
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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(Decoder, self).__init__()

        self.device = device
        self.tokenizer = tokenizer
        self.embed = nn.Embedding(tokenizer.vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward_step(
        self, input_token_id: Tensor, states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        embedding = self.embed(input_token_id)
        embedding = self.dropout(embedding)

        print(states[0].shape)
        print(embedding.shape)
        output, states = self.lstm(embedding, states)
        logits = self.linear(output)
        print(logits.shape)

        return logits, states

    def forward(
        self,
        states: Tuple[Tensor, Tensor],
        target_token_ids: Union[Tensor, None] = None,
        max_len: int = 50,
    ) -> Tensor:
        batch_size = states[0].shape[0]
        input_token_id = (
            torch.empty(batch_size, 1, dtype=torch.int64)
            .fill_(self.tokenizer.convert_tokens_to_ids("[BOS]"))
            .to(self.device)
        )
        print(input_token_id.shape)

        logits_list = []

        for i in range(max_len):
            # shape: (1, batch_size, hidden_size)
            states = states[0].unsqueeze(0), states[1].unsqueeze(0)

            logits, states = self.forward_step(input_token_id, states)

            logits_list.append(logits)
            output_token_id = logits.argmax(1)

            if target_token_ids is not None:
                input_token_id = target_token_ids[:, i]
            else:
                input_token_id = output_token_id.detach()

            ids = output_token_id.tolist()
            print(ids)
            tokens = self.tokenizer.convert_ids_to_tokens()

            if all(token == "[EOS]" for token in tokens):
                break

        return torch.stack(tuple(logits_list))


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device = torch.device("cpu"),
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

        features = feature_maps.view(feature_maps.shape[0], feature_maps.shape[1], -1)
        states = (features.mean(2), features.mean(2))

        logits_tensor: Tensor = self.decoder(states, target_token_ids, max_len)
        return logits_tensor
