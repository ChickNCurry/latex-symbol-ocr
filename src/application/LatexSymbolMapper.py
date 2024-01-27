from typing import List

import pandas as pd

from src.application.models import IMapper


class LatexSymbolMapper(IMapper):
    def __init__(self) -> None:
        self._markup_df = pd.read_csv("src/application/state/symbols.csv")

    def map_to_markup(self, class_labels: List[int]) -> List[str]:
        markups = [
            self._markup_df[self._markup_df["symbol_id"] == c].iloc[0]["latex"]
            for c in class_labels
        ]
        return markups
