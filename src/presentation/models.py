from dataclasses import dataclass
from tkinter.ttk import Button, Entry, Label


@dataclass
class PredictionComponent:
    label_ranking: Label
    label_render: Label
    entry_markup: Entry
    button_copy: Button
