import torch
from torch import Tensor
import nltk
from transformers import PreTrainedTokenizerFast


def get_bleu(
    y_pred: Tensor, y_true: Tensor, tokenizer: PreTrainedTokenizerFast
) -> float:
    # print(y_pred.shape, y_true.shape)

    bleus = []

    for i in range(y_pred.shape[0]):
        ids_pred = torch.argmax(y_pred[i], dim=1).tolist()
        ids_true = torch.argmax(y_true[i], dim=1).tolist()

        # print(ids_pred)
        # print(ids_true)

        tokens_pred = tokenizer.convert_ids_to_tokens(ids_pred)
        tokens_true = tokenizer.convert_ids_to_tokens(ids_true)

        # print(tokens_pred)
        # print(tokens_true)

        bleu = nltk.translate.bleu_score.sentence_bleu(tokens_true, tokens_pred)

        # print(bleu)

        bleus.append(bleu)

    avg = sum(bleus) / len(bleus)

    return avg


def print_statistics(
    epoch: int, batch: int, num_batches: int, loss: float, bleu: float
) -> None:
    print(
        f"EPOCH {epoch + 1} | BATCH {batch + 1} of {num_batches} | LOSS {loss:.4f} | BLEU {bleu:.4f}"
    )
