from typing import List

import pandas as pd

from src.application.interfaces import IMapper


class LatexSymbolMapper(IMapper):
    def __init__(self) -> None:
        self.markup_df = pd.read_csv("src/application/state/symbols.csv")

    def map_to_markup(self, class_labels: List[int]) -> List[str]:
        markups = []
        for c in class_labels:
            markup = self.markup_df[self.markup_df["symbol_id"] == c].iloc[0]["latex"]
            markups.append(markup)
        return markups
